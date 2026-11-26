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

The state may hold further unknowns, recognised by name: a field ``por`` is
the porosity, and a scalar named after a field of ``minires.fluids.Fluid``
(``sor``, ``swc``, ``vo``, ``vw``, ``no``, ``nw``, ...) overrides that
parameter of the fluid that ``[simulator.model]`` describes.

For optimization the wells can be *controlled* by the state instead: ``controls``
maps a state variable to the rate (or BHP) of a well,

    [simulator.controls]
        rate_inj1 = {well = "INJ1", type = "rate"}

whose values, one per control interval, are held constant over equally long
intervals of the run. With ``controls`` the permeability need not be in the
state; the ``K`` of ``[simulator.model]`` is then the reservoir's.

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

DIFFERENTIABLE = ("WOPR", "WWPR", "WWCT", "WBHP")
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
        - ``p0``: initial pressure, a scalar or a field (default ``0``). Only
          consequential if the model's ``ct > 0``.
        - ``compute_adjoints``: also return each datum's sensitivity to the
          state (default ``False``), ref :meth:`adjoint_frame`.
        - ``controls``: state variables that set a well's schedule, as
          ``{variable: {"well": name, "type": "rate" | "bhp", "sign": 1}}``
          (``type`` defaults to ``"rate"``; rates are positive for injection, so
          give ``sign = -1`` to control a production rate by its magnitude). The
          variable's values, whose number must divide the number of steps, are
          held constant over equally long intervals. Default: none, and the
          wells run as ``model`` gives them. With controls the permeability may
          be left out of the state, the ``K`` of ``model`` then being used, and
          an ``aux_input`` in the state (robust optimization) is ignored.
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
        self.well_records = cfg.get("model", {}).get("wells", [])
        self.S0 = np.full(self.model.Nxy, 0.0) + np.ravel(cfg.get("s0", 0.0))
        self.P0 = np.full(self.model.Nxy, 0.0) + np.ravel(cfg.get("p0", 0.0))
        self._parse_datatypes()
        self._parse_controls(cfg.get("controls", None))

    def _parse_controls(self, controls) -> None:
        """Check the ``controls`` against the model: ``{variable: (index of the well's record, type, sign)}``."""
        self.controls = {}
        if not controls:
            return
        if self.compute_adjoints:
            raise NotImplementedError("compute_adjoints is on, but the adjoint is not with respect to well controls.")
        names = [record.get("name") for record in self.well_records]
        for variable, spec in controls.items():
            well, kind, sign = spec.get("well"), spec.get("type", "rate"), float(spec.get("sign", 1.0))
            if well not in names:
                raise ValueError(f"The control '{variable}' names the well '{well}'; the model's wells are {names}.")
            if kind not in ("rate", "bhp"):
                raise ValueError(f"The control '{variable}' has type '{kind}'; it must be 'rate' or 'bhp'.")
            self.controls[variable] = (names.index(well), kind, sign)

    def controlled_wells(self, state: dict) -> list:
        """The well records with the schedules that the member's controls give them."""
        records = deepcopy(self.well_records)
        for variable, (index, kind, sign) in self.controls.items():
            if variable not in state:
                continue  # the well runs as the model gives it
            values = np.asarray(state[variable], dtype=float).ravel()
            if values.size == 0 or self.nSteps % values.size:
                raise ValueError(
                    f"The control '{variable}' has {values.size} values, which does not divide the "
                    f"{self.nSteps} steps of the run into equal intervals."
                )
            record = records[index]
            record.pop("bhp" if kind == "rate" else "rate", None)  # a well is on one or the other
            record[kind] = sign * np.repeat(values, self.nSteps // values.size)
        return records

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
        try:
            model, SS, PP = self._simulate(state)
        except Exception:
            logger.exception("MiniRes failed on member %s; it will be replaced.", member_index)
            return False

        records = self.records(model, SS)
        if self.compute_adjoints:
            return records, self.adjoint_frame(model, SS, PP)
        return records

    def set_state(self, model, state: dict) -> None:
        """Write the member's state into the model: the permeability, and whatever else it holds."""
        fluid_parameters = {f.name for f in dataclass_fields(model.fluid)}
        others = set(state) - {self.state_variable}
        if self.controls:
            others -= set(self.controls) | {"aux_input"}
        unknown = others - {"por"} - fluid_parameters
        if unknown:
            raise KeyError(
                f"The state has {sorted(unknown)}, which MiniRes has no use for. Besides the permeability "
                f"('{self.state_variable}') it takes 'por' and the fluid's {sorted(fluid_parameters)}."
            )
        if self.compute_adjoints and others:
            raise NotImplementedError(
                f"compute_adjoints is on, but the state also has {sorted(others)}; "
                "the adjoint is only with respect to the permeability."
            )

        if self.controls:
            wells = self.controlled_wells(state)
            if self.state_variable in state:
                self.set_permeability(model, state, wells)
            else:
                model.wells = wells  # on the permeability of the model
        else:
            self.set_permeability(model, state)
        if "por" in state:
            model.por = self.as_field(model, "por", state["por"])
        for name in sorted(others & fluid_parameters):
            value = np.asarray(state[name], dtype=float)
            if value.size != 1:
                raise ValueError(f"'{name}' is a parameter of the fluid, a scalar, but has {value.size} values.")
            setattr(model.fluid, name, value.item())

    def set_permeability(self, model, state: dict, wells=None) -> None:
        """Write the member's field into the model, as its (isotropic) permeability, and assemble ``wells`` on it."""
        if self.state_variable not in state:
            raise KeyError(
                f"The state has no '{self.state_variable}' (it has {sorted(state)}). "
                f"Name the permeability state with 'state_variable' in the simulator config."
            )
        field = self.as_field(model, self.state_variable, state[self.state_variable])
        model.K = np.exp(field) if self.log_perm else field
        # The wells' indices (Peaceman's, for those given an `rw`) are derived from K
        # when the wells are assembled, so they are assembled again for this K.
        model.wells = self.well_records if wells is None else wells

    def as_field(self, model, name: str, values) -> np.ndarray:
        """The state vector ``values`` as a field on the model's grid, in its declared ordering."""
        field = np.asarray(values, dtype=float).ravel()
        if field.size != model.Nxy:
            raise ValueError(f"'{name}' has {field.size} values, but the grid has {model.Nxy} cells.")
        return field.reshape(model.shape, order=self.field_order)

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

        The quantities covered are those of :data:`DIFFERENTIABLE`, on either
        control: MiniRes differentiates the states, and the well model that
        turns them into rates and bottom-hole pressures is differentiated here
        (:meth:`_add_seed`), as is the well indices' dependence on the
        permeability (:meth:`_state_gradient`).
        """
        seeds = self._seed_buffers(model, SS, PP)
        rows = []
        for step in self.report:
            row = {}
            for dtype, (quantity, well) in zip(self.datatype, self._parsed):
                for buf in seeds:
                    buf.fill(0)
                self._add_seed(model, SS, PP, dtype, quantity, well, step, 1.0, seeds)
                row[dtype] = self._state_gradient(model, SS, PP, *seeds)
            rows.append(row)

        frame = pd.DataFrame(rows, index=self.true_order[1])
        frame.index.name = self.report_type
        return frame

    def run_vjp(self, state: dict, weights) -> dict:
        """Weighted sums of the data's sensitivities to the state: ``{variable: (k, n)}``.

        ``weights`` is a frame indexed as the records are, report points by
        data types; each cell holds the ``(k,)`` weights of that datum in the
        ``k`` sums, or ``None`` for a datum that is in none of them. For the
        permeability the cost is one simulation and then one backward sweep per
        sum, whatever the number of data, since the seeds of the adjoint add
        up. A scalar of the fluid in the state costs two more simulations.
        """
        fluid_parameters = {f.name for f in dataclass_fields(self.model.fluid)}
        scalars = sorted(set(state) - {self.state_variable})
        if set(scalars) - fluid_parameters:
            raise NotImplementedError(
                f"The state also has {sorted(set(scalars) - fluid_parameters)}; the sensitivity is only "
                "with respect to the permeability and the scalars of the fluid."
            )
        position = {label: i for i, label in enumerate(self.true_order[1])}
        parsed = dict(zip(self.datatype, self._parsed))
        cells = [(label, dtype, np.ravel(w)) for label, row in weights.iterrows() for dtype, w in row.items()
                 if w is not None]

        model, SS, PP = self._simulate(state)
        seeds = self._seed_buffers(model, SS, PP)
        directions = []
        for j in range(len(cells[0][2])):
            for buf in seeds:
                buf.fill(0)
            for label, dtype, w in cells:
                if w[j]:
                    self._add_seed(model, SS, PP, dtype, *parsed[dtype], self.report[position[label]], w[j], seeds)
            directions.append(self._state_gradient(model, SS, PP, *seeds))
        products = {self.state_variable: np.asarray(directions)}

        def sums(perturbed):
            model, SS, _ = self._simulate(perturbed)
            records = self.records(model, SS)
            return sum(w * records[position[label]][dtype][0] for label, dtype, w in cells)

        for name in scalars:
            # shortcut: a central difference, at two simulations per scalar. The step is wide
            # because the response is only piecewise smooth (the number of transport sub-steps
            # follows the CFL number, which the fluid sets): a narrow one measures the jumps. That
            # leaves an error of some 10 %. An exact derivative needs MiniRes's adjoint to
            # differentiate the relative permeabilities with respect to their parameters.
            value = float(np.ravel(state[name])[0])
            h = 1e-2 * max(1.0, abs(value))
            plus, minus = (sums({**state, name: np.array([value + d])}) for d in (h, -h))
            products[name] = ((plus - minus) / (2 * h))[:, None]
        return products

    def _simulate(self, state: dict) -> tuple:
        """Run the member on its own copy of the model: ``(model, SS, PP)``."""
        model = deepcopy(self.model)
        self.set_state(model, state)
        SS, PP = model.sim(self.dt, self.nSteps, self.S0, self.P0, pbar=False)
        return model, SS, PP

    def _seed_buffers(self, model, SS, PP):
        wells = model.wells
        if any("aquifer" in record for record in self.well_records):
            raise NotImplementedError("The adjoint does not cover a model with an aquifer.")
        for i, record in enumerate(self.well_records):
            if "rw" not in record or "WI" in record or wells.group is None:
                continue
            comps = np.flatnonzero(np.asarray(wells.group) == i)
            if len(comps) > 1 and (wells.bhp is None or not np.isfinite(wells.bhp[comps]).all()):
                raise NotImplementedError("The adjoint does not differentiate rate allocation for "
                                          f"rate-controlled multi-completion well {wells.names[i]}.")
        return np.zeros_like(SS), np.zeros_like(PP), np.zeros(wells.nComp)

    def _add_seed(self, model, SS, PP, dtype, quantity, well, step, weight, seeds):
        """Accumulate one weighted datum into ``(dS, dP, dlogWI)``.

        ``dS`` and ``dP``, shaped as ``SS`` and ``PP``, are the seeds of
        ``minires.tlm.adjoint``; ``dlogWI`` is with respect to the log of the
        completions' well indices, ``(nComp,)``. The well operates over a step
        on the saturation at its start and the pressure at its end, with a rate
        of ``WI λ_t (p_bh - p_cell)``; the water cut is of the saturation at the
        end.
        """
        if quantity not in DIFFERENTIABLE:
            raise NotImplementedError(
                f"'{dtype}' is not among the differentiated quantities {DIFFERENTIABLE}. "
                "Seed minires.tlm.adjoint by hand for others."
            )
        wells = model.wells
        group = np.arange(wells.nComp) if wells.group is None else np.asarray(wells.group)
        cells = model.xy2ind(wells.xy[:, 0], wells.xy[:, 1])
        k = step - 1
        rates = np.asarray(wells.actual_rates)[:, k]
        on_bhp = np.isfinite(wells.at_time("bhp", np.nan, k))
        mobility = np.sum(model.fluid.RelPerm(SS[k][cells]), axis=0)
        dlog_mobility = np.sum(model.fluid.dRelPerm(SS[k][cells]), axis=0) / mobility
        dS, dP, dlogWI = seeds

        if quantity == "WBHP":
            i = np.flatnonzero(group == well)[0]  # a wellbore's completions share one BHP
            if not on_bhp[i]:  # on BHP control it is the control: a constant
                drawdown = np.asarray(wells.actual_bhp)[i, k] - PP[step][cells[i]]  # rate / (WI λ_t)
                if not np.isfinite(drawdown):
                    raise NotImplementedError(f"'{dtype}': the well has no well index, hence no BHP.")
                dP[step, cells[i]] += weight
                dS[k, cells[i]] -= weight * drawdown * dlog_mobility[i]
                dlogWI[i] -= weight * drawdown
            return

        comps = np.flatnonzero(group == well)
        produced = np.maximum(-rates[comps], 0.0)
        fw = model.fluid.fractional_flow(SS[step][cells[comps]])
        liquid = produced.sum()
        # The datum's derivatives with respect to each completion's produced rate and water cut
        if quantity == "WWPR":
            d_rate, d_cut = fw, produced
        elif quantity == "WOPR":
            d_rate, d_cut = 1.0 - fw, -produced
        elif liquid > 0:  # WWCT
            d_rate, d_cut = (fw - produced @ fw / liquid) / liquid, produced / liquid
        else:
            d_rate, d_cut = 0.0 * fw, 0.0 * fw
        np.add.at(dS[step], cells[comps], weight * d_cut * model.fluid.dfractional_flow(SS[step][cells[comps]]))

        # On rate control the produced rate is given; on BHP control it follows from the state
        free = comps[on_bhp[comps] & (rates[comps] < 0)]
        d_rate = weight * d_rate[np.isin(comps, free)] * -rates[free]
        np.add.at(dS[k], cells[free], d_rate * dlog_mobility[free])
        np.add.at(dP[step], cells[free], d_rate / (PP[step][cells[free]] - np.asarray(wells.actual_bhp)[free, k]))
        dlogWI[free] += d_rate

    def _state_gradient(self, model, SS, PP, dS: np.ndarray, dP: np.ndarray, dlogWI: np.ndarray) -> np.ndarray:
        """The backward sweep from the seeds, as a gradient with respect to the state vector."""
        from minires import tlm

        wells = model.wells
        grad = tlm.adjoint(model, self.dt, SS, PP, dS, dP)
        dlogK = grad.logK.sum(0).ravel()  # isotropic: the state feeds both components

        # MiniRes holds the well indices fixed, but those derived from K (Peaceman's, of the wells
        # given an `rw`) are proportional to it. A well index only enters as WI λ_t (p_bh - p_cell),
        # as the BHP control does, so its gradient follows from that of the control.
        group = np.arange(wells.nComp) if wells.group is None else np.asarray(wells.group)
        cells = model.xy2ind(wells.xy[:, 0], wells.xy[:, 1])
        drawdown = np.asarray(wells.actual_bhp) - PP[1:, cells].T
        dlogWI = dlogWI + np.nansum(grad.bhp * drawdown, axis=1)
        from_K = np.array([("rw" in record or "path" in record) and "WI" not in record
                           for record in self.well_records], dtype=bool)[group]
        np.add.at(dlogK, cells[from_K], dlogWI[from_K])

        dlogK = dlogK.reshape(model.shape)
        if not self.log_perm:
            dlogK = dlogK / model.K[0]
        return dlogK.ravel(order=self.field_order)
