"""Forward simulator backed by MiniRes, a two-phase (water/oil) TPFA reservoir simulator.

MiniRes (https://github.com/patnr/MiniRes) is a pure-Python toy simulator: no
binary, no licence, no deck, no scratch folder. It is therefore the cheapest
way to drive PET with actual two-phase flow -- in a tutorial, in CI, or as the
reference case when developing a scheme -- rather than with an ODE.

Install it alongside PET with ``pip install PET[minires]``.

The simulator section of the config configures it, e.g.

    [simulator]
        reporttype  = "steps"
        reportpoint = [2, 4, 6, 8, 10]
        datatype    = ["WWCT:PRD1", "WOPR:PRD1", "FWIR"]
        dt          = 0.025
        parallel    = 1

        [simulator.model]
            Nx = 32
            Ny = 32
            por = 0.2
            [[simulator.model.wells]]
                name = "INJ1"
                xy = [0.1, 0.1]
                rate = 1.0
            [[simulator.model.wells]]
                name = "PRD1"
                xy = [0.9, 0.9]
                rate = -1.0

``[simulator.model]`` is passed to ``minires.ResSim`` as it stands (its
``wells`` are MiniRes well records), less the permeability, which is what the
ensemble state supplies, one field per member.

Conventions, all of which this module owns -- MiniRes itself is agnostic:

- **Report points** are *step indices*, ``1 .. nSteps``, since MiniRes takes a
  uniform ``dt``. Use ``reporttype = "steps"``. A dated case can map them with
  ``reportdates``.
- **Field ordering.** MiniRes is C-major (x is the first axis); Eclipse, and
  hence most of PET's tooling, is Fortran-ordered. ``field_order`` (default
  ``"C"``) says which the *state vector* is in, and is applied on the way in
  and on the way out (the adjoint).
- **Data types** are named as Eclipse's summary vectors, ``<quantity>:<well>``
  for a well and ``<quantity>`` for the field, so observed-data files,
  ``datatype`` filters and localization tooling need no adaptation. Rates are
  positive as produced/injected, and areal (MiniRes has no thickness).
- **Units** are whatever the config poses the model in. Set ``cdarcy = 0.008527``
  in ``[simulator.model]`` for metric (m, day, bar, mD, cP), as Eclipse does.
"""

import logging
from copy import deepcopy
from dataclasses import fields as dataclass_fields

import numpy as np
import pandas as pd

__all__ = ["MiniRes"]

logger = logging.getLogger(__name__)

WELL_QUANTITIES = ("WOPR", "WWPR", "WLPR", "WWIR", "WWCT", "WBHP")
"""Per-well quantities: oil/water/liquid production rate, water injection rate, water cut, BHP."""

FIELD_QUANTITIES = ("FOPR", "FWPR", "FWIR", "FOPT", "FWPT", "FWIT")
"""Field rates and their cumulatives."""

DIFFERENTIABLE = ("WOPR", "WWPR", "WWCT")
"""The quantities :meth:`MiniRes.run_fwd_sim` can also produce an adjoint for."""


def build_model(spec: dict):
    """Build the ``minires.ResSim`` that ``[simulator.model]`` describes.

    ``por`` and ``active`` may be given as scalars; the wells are MiniRes well
    records (``xy``/``path``, ``rate``/``bhp``, ``rw``/``WI``, ``name``).
    """
    from minires import ResSim

    spec = deepcopy(dict(spec))
    known = {f.name for f in dataclass_fields(ResSim)}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in [simulator.model]: {sorted(unknown)}. "
            f"The section is passed to minires.ResSim, whose parameters are {sorted(known)}."
        )

    # K is the state, not a config item: whatever is given here is only the shape's stand-in.
    model = ResSim(**{k: v for k, v in spec.items() if k not in ("por", "active", "wells")})
    for key in ("por", "active"):
        if key in spec:  # ResSim broadcasts a scalar K, but not these
            val = spec[key]
            setattr(model, key, np.full(model.shape, val) if np.isscalar(val) else val)
    model.wells = spec.get("wells", [])
    return model


class MiniRes:
    """PET forward simulator: one MiniRes run per ensemble member.

    Satisfies :class:`ensemble.protocols.ForwardSimulator`. Every member runs
    on its own ``deepcopy`` of the model, so nothing is shared and the class is
    picklable -- which is what ``parallel > 1`` (``p_map``) requires. MiniRes
    drops its cached pressure preconditioner on copy for exactly this reason.

    Parameters
    ----------
    input_dict : dict
        The parsed ``[simulator]`` section. Keys:

        - ``dt``: the time step. Required.
        - ``reportpoint``: the step indices to report at. Required.
        - ``reporttype``: the index's name (default ``"steps"``).
        - ``reportdates``: optional labels to report *under* instead of the step
          indices, e.g. dates, one per report point.
        - ``datatype``: the summary vectors to report, ref the module docstring.
        - ``model``: what :func:`build_model` takes.
        - ``state_variable``: the ensemble state that supplies the permeability
          (default ``"permx"``).
        - ``log_perm``: whether that state is :math:`\\log K` (default ``True``).
        - ``field_order``: the state vector's grid ordering, ``"C"`` (default)
          or ``"F"``.
        - ``s0``: initial water saturation, a scalar or a field (default ``0``).
        - ``compute_adjoints``: also return each datum's sensitivity to the
          state (default ``False``), ref :meth:`adjoint_frame`.
        - ``levels``: per-fidelity overrides of the above, for multilevel runs,
          selected by ``setup_fwd_run(level=...)``.
        - ``parallel``, ``hpc``: read by the ensemble, not by this class.
    """

    def __init__(self, input_dict: dict):
        self.input_dict = input_dict
        self.levels = input_dict.get("levels", None)
        self._configure(input_dict)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _configure(self, cfg: dict) -> None:
        """Internalize one configuration -- the whole of it, or one fidelity level's."""
        for required in ("dt", "reportpoint", "datatype"):
            assert required in cfg, f"'{required}' is missing from the simulator config"

        self.dt = float(cfg["dt"])
        self.report = [int(r) for r in cfg["reportpoint"]]
        self.report_type = cfg.get("reporttype", "steps")
        self.nSteps = max(self.report)

        # The label the records are indexed by: the step, or what the config renames it to
        labels = cfg.get("reportdates", self.report)
        assert len(labels) == len(self.report), "'reportdates' must have one label per report point"
        self.true_order = [self.report_type, list(labels)]
        self.true_prim = self.true_order
        self.l_prim = list(range(len(self.report)))

        self.datatype = list(cfg["datatype"])
        self.all_data_types = self.datatype
        self.compute_adjoints = bool(cfg.get("compute_adjoints", False))

        self.state_variable = cfg.get("state_variable", "permx")
        self.log_perm = bool(cfg.get("log_perm", True))
        self.field_order = cfg.get("field_order", "C")
        assert self.field_order in ("C", "F"), "'field_order' must be 'C' or 'F'"

        self.model = build_model(cfg.get("model", {}))
        self.S0 = np.full(self.model.Nxy, 0.0) + np.ravel(cfg.get("s0", 0.0))
        self._parse_datatypes()

    def _parse_datatypes(self) -> None:
        """Split each data type into ``(quantity, well)``, checking it against the model."""
        names = list(self.model.wells.names or [])
        self._parsed = []
        for dtype in self.datatype:
            quantity, _, well = dtype.partition(":")
            if well:
                if quantity not in WELL_QUANTITIES:
                    raise ValueError(f"Unknown well quantity '{quantity}' in '{dtype}'. Known: {WELL_QUANTITIES}.")
                if well not in names:
                    raise ValueError(f"'{dtype}' names no well of the model. Its wells are {names}.")
                self._parsed.append((quantity, names.index(well)))
            else:
                if quantity not in FIELD_QUANTITIES:
                    raise ValueError(f"Unknown field quantity '{quantity}'. Known: {FIELD_QUANTITIES}.")
                self._parsed.append((quantity, None))

        if self.compute_adjoints:
            bad = [d for d, (q, w) in zip(self.datatype, self._parsed) if q not in DIFFERENTIABLE]
            if bad:
                raise NotImplementedError(
                    f"compute_adjoints is on, but {bad} are not among the differentiated "
                    f"quantities {DIFFERENTIABLE}. Seed minires.tlm.adjoint by hand for others."
                )

    def setup_fwd_run(self, level=None, **kwargs) -> None:
        """Select the fidelity ``level``'s configuration, when the config gives ``levels``."""
        if self.levels is None or level is None:
            return
        cfg = {**self.input_dict, **self.levels[level]}
        self._configure(cfg)

    # ------------------------------------------------------------------
    # One member
    # ------------------------------------------------------------------
    def run_fwd_sim(self, state: dict, member_index: int = 0, **kwargs):
        """Simulate one realisation; return its records (and adjoint), or ``False`` if it failed."""
        model = deepcopy(self.model)
        try:
            self.set_permeability(model, state)
            SS, PP = model.sim(self.dt, self.nSteps, self.S0, pbar=False)
        except Exception:
            logger.exception("MiniRes failed on member %s; it will be replaced.", member_index)
            return False

        records = self.records(model, SS)
        if self.compute_adjoints:
            return records, self.adjoint_frame(model, SS, PP)
        return records

    def set_permeability(self, model, state: dict) -> None:
        """Write the member's field into the model, as its (isotropic) permeability."""
        if self.state_variable not in state:
            raise KeyError(
                f"The state has no '{self.state_variable}' (it has {sorted(state)}). "
                f"Name the permeability state with 'state_variable' in the simulator config."
            )
        field = np.asarray(state[self.state_variable], dtype=float).ravel()
        if field.size != model.Nxy:
            raise ValueError(f"'{self.state_variable}' has {field.size} values, but the grid has {model.Nxy} cells.")
        field = field.reshape(model.shape, order=self.field_order)
        model.K = np.exp(field) if self.log_perm else field

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def well_report(self, model, SS: np.ndarray) -> dict:
        """Every reportable quantity, as ``(nWell or 1, nSteps)`` arrays, from the run's well operation.

        MiniRes reports one *total* rate per completion (signed: positive
        injects). The phase split is the cell's fractional flow -- the same
        ``Fluid.fractional_flow`` the transport uses -- and the completions are
        summed into wells by ``Wells.group``.
        """
        wells = model.wells
        cells = model.xy2ind(wells.xy[:, 0], wells.xy[:, 1])
        rates = np.asarray(wells.actual_rates)                     # (nComp, nSteps)
        fw = model.fluid.fractional_flow(SS[1:][:, cells]).T       # (nComp, nSteps), end of step
        produced = np.where(rates < 0, -rates, 0.0)
        injected = np.where(rates > 0, rates, 0.0)

        group = np.arange(wells.nComp) if wells.group is None else np.asarray(wells.group)

        def by_well(per_completion):
            out = np.zeros((wells.nWell, per_completion.shape[1]))
            np.add.at(out, group, per_completion)
            return out

        report = {
            "WWPR": by_well(produced * fw),
            "WOPR": by_well(produced * (1 - fw)),
            "WLPR": by_well(produced),
            "WWIR": by_well(injected),
        }
        with np.errstate(invalid="ignore", divide="ignore"):
            report["WWCT"] = np.where(report["WLPR"] > 0, report["WWPR"] / report["WLPR"], 0.0)
        # A wellbore's completions share one BHP, so the first of each well's is the well's
        first = np.zeros(wells.nWell, int)
        first[group[::-1]] = np.arange(wells.nComp)[::-1]
        report["WBHP"] = np.asarray(wells.actual_bhp)[first]

        for field, well in (("FOPR", "WOPR"), ("FWPR", "WWPR"), ("FWIR", "WWIR")):
            report[field] = report[well].sum(0, keepdims=True)
        for cum, rate in (("FOPT", "FOPR"), ("FWPT", "FWPR"), ("FWIT", "FWIR")):
            report[cum] = np.cumsum(report[rate], axis=1) * self.dt
        return report

    def records(self, model, SS: np.ndarray) -> list:
        """One dict per report point, keyed by data type -- what the ensemble collects."""
        report = self.well_report(model, SS)
        out = []
        for step in self.report:
            row = {}
            for dtype, (quantity, well) in zip(self.datatype, self._parsed):
                series = report[quantity]
                row[dtype] = np.array([series[0 if well is None else well, step - 1]])
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # Adjoints
    # ------------------------------------------------------------------
    def adjoint_frame(self, model, SS: np.ndarray, PP: np.ndarray) -> pd.DataFrame:
        """Each datum's sensitivity to the state, as the frame the ensemble stacks into ``(nd, nx, ne)``.

        MiniRes's adjoint (``minires.tlm``) gives the gradient of *one* scalar
        per backward sweep, at about the cost of one simulation -- so a full
        Jacobian costs one sweep per datum. That is affordable on the grids this
        simulator is for, and is what PET's adjoint-based analyses want.

        Only quantities that are a function of the saturation at the well's
        cells are covered (:data:`DIFFERENTIABLE`), and only at rate-controlled
        completions, whose rate is then a constant of the objective. A
        BHP-controlled well's rate is itself a function of ``(S, P)``; that
        derivative has to be worked into the seed by hand, so it is refused
        rather than silently dropped.
        """
        from minires import tlm

        wells = model.wells
        cells = model.xy2ind(wells.xy[:, 0], wells.xy[:, 1])
        group = np.arange(wells.nComp) if wells.group is None else np.asarray(wells.group)
        rates = np.asarray(wells.actual_rates)
        report = self.well_report(model, SS)

        rows = []
        for step in self.report:
            row = {}
            for dtype, (quantity, well) in zip(self.datatype, self._parsed):
                seed = np.zeros((self.nSteps + 1, model.Nxy))
                k = step  # the datum is the end of step `step`, i.e. SS[step]
                for comp in np.flatnonzero(group == well):
                    if np.isfinite(wells.at_time("bhp", np.nan, min(k, self.nSteps) - 1)[comp]):
                        raise NotImplementedError(
                            f"'{dtype}' is on BHP control; its rate is itself a function of the state. "
                            "Seed minires.tlm.adjoint by hand for it."
                        )
                    s = SS[k][cells[comp]]
                    dfw = model.fluid.dfractional_flow(np.array([s]))[0]
                    produced = max(-rates[comp, k - 1], 0.0)
                    if quantity == "WWPR":
                        seed[k, cells[comp]] += produced * dfw
                    elif quantity == "WOPR":
                        seed[k, cells[comp]] -= produced * dfw
                    elif quantity == "WWCT":
                        liquid = report["WLPR"][well, k - 1]
                        seed[k, cells[comp]] += produced * dfw / liquid if liquid > 0 else 0.0

                grad = tlm.adjoint(model, self.dt, SS, PP, seed)
                dlogK = grad.logK.sum(0)  # isotropic: the state feeds both components
                if not self.log_perm:
                    dlogK = dlogK / model.K[0]
                row[dtype] = dlogK.ravel(order=self.field_order)
            rows.append(row)

        frame = pd.DataFrame(rows, index=self.true_order[1])
        frame.index.name = self.report_type
        return frame
