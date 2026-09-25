"""
Simulator wrapper for the Van der Pol oscillator.
Van der Pol oscillator is a non-conservative oscillator with non-linear damping.

The equation of motion is given by:

    x'' - μ(1 - x^2)x' + x = 0

where μ is a scalar parameter indicating the nonlinearity and the strength of the damping.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from multiprocessing import Pool


__author__ = "copilot, Mathias Methlie Nilsen, Andreas Stordal"
__all__ = ["VanDerPolOscillator", "_integrate"]


# ---------------------------------------------------------------------------
# ODE definition
# ---------------------------------------------------------------------------

def _vdp_rhs(t, state, mu):
    """Van der Pol ODE augmented with first-order sensitivity equations."""
    x1, x2 = state[0], state[1]

    # Sensitivity states
    S11, S12, S13 = state[2], state[3], state[4]   # dx1/d[x1_0, x2_0, mu]
    S21, S22, S23 = state[5], state[6], state[7]   # dx2/d[x1_0, x2_0, mu]

    # Jacobian of f w.r.t. state
    A11 =  0.0
    A12 =  1.0
    A21 = -2.0 * mu * x1 * x2 - 1.0
    A22 =  mu * (1.0 - x1 ** 2)

    # Derivative of f w.r.t. mu
    B1 = 0.0
    B2 = (1.0 - x1 ** 2) * x2

    # State dynamics
    dx1 = x2
    dx2 = mu * (1.0 - x1 ** 2) * x2 - x1

    # Sensitivity dynamics  dS/dt = A @ S + B  (column-wise)
    dS11 = A11 * S11 + A12 * S21
    dS12 = A11 * S12 + A12 * S22
    dS13 = A11 * S13 + A12 * S23 + B1

    dS21 = A21 * S11 + A22 * S21
    dS22 = A21 * S12 + A22 * S22
    dS23 = A21 * S13 + A22 * S23 + B2

    return [dx1, dx2, dS11, dS12, dS13, dS21, dS22, dS23]


def _integrate(x1_0, x2_0, mu, t_eval, atol=1e-5, rtol=1e-5):
    """
    Integrate the Van der Pol system (with sensitivities) for one member.

    Returns
    -------
    sol_T : ndarray, shape (len(t_eval), 8)
        Columns: [x1, x2, S11, S12, S13, S21, S22, S23]
    """
    state0 = [x1_0, x2_0,
              1.0, 0.0, 0.0,   # S11, S12, S13
              0.0, 1.0, 0.0]   # S21, S22, S23

    sol = solve_ivp(
        _vdp_rhs,
        [t_eval[0], t_eval[-1]],
        state0,
        args=(mu,),
        t_eval=t_eval,
        method="RK45",
        atol=atol,
        rtol=rtol,
    )
    return sol.y.T  # (n_times, 8)


# ---------------------------------------------------------------------------
# Worker function (must be module-level for multiprocessing)
# ---------------------------------------------------------------------------

def _run_single(args):
    """Run a single ensemble member; used by the parallel pool."""
    member_input, idn, t_eval, datatypes, compute_adjoints, atol, rtol = args

    x1_0 = float(member_input.get("x1", 1.0))
    x2_0 = float(member_input.get("x2", 0.0))
    mu   = float(member_input.get("mu", 1.0))

    sol = _integrate(x1_0, x2_0, mu, t_eval, atol=atol, rtol=rtol)

    # ------------------------------------------------------------------
    # Build output list: one dict per reportpoint
    # ------------------------------------------------------------------
    _state_col = {"x1": 0, "x2": 1}
    result = []
    for it in range(sol.shape[0]):
        row = {}
        for key in datatypes:
            col = _state_col.get(key)
            if col is None:
                raise ValueError(f"Unknown datatype '{key}'. Supported: 'x1', 'x2'.")
            row[key] = float(sol[it, col])
        result.append(row)

    if not compute_adjoints:
        return result

    # ------------------------------------------------------------------
    # Sensitivity matrix  dY/d[x1_0, x2_0, mu]
    # Shape: (n_obs_total, 3)
    # Sensitivity columns in sol:  S11=2, S12=3, S13=4  (for x1)
    #                               S21=5, S22=6, S23=7  (for x2)
    # ------------------------------------------------------------------
    _sens_cols = {"x1": [2, 3, 4], "x2": [5, 6, 7]}
    sens = {}
    for key in datatypes:
        cols = _sens_cols[key]
        sens[key] = sol[:, cols].copy()     # (n_times, 3)

    return result, sens


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class VanDerPolOscillator:
    """
    PET-compatible wrapper for the Van der Pol oscillator.

    Parameters
    ------------
    options : dict
        Configuration options for the simulator. Supported keys:
        - ``reportpoint``: list of report points (default: [1, 2, ..., 15])
        - ``reporttype``: type of report points, e.g. "times" (default: "times")
        - ``datatype``: list of datatypes to extract, e.g. ["x1", "x2"] (default: ["x1"])
        - ``compute_adjoints``: bool, whether to compute adjoints (default: False)
        - ``atol``: absolute tolerance for ODE solver (default: 1e-5)
        - ``rtol``: relative tolerance for ODE solver (default: 1e-5)
        - ``parallel``: number of parallel processes to use (default: 1, i.e. no parallelism)
    """

    def __init__(self, options: dict):
        # Report / index
        self.report      = options.get("reportpoint", list(range(1, 16)))
        self.report_type = options.get("reporttype", "times")
        self.index       = [self.report_type, self.report]

        # Datatypes to extract
        self.datatype = options.get("datatype", ["x1"])

        # Adjoint flag
        self.compute_adjoints = options.get("compute_adjoints", False)

        # Solver tolerances
        self.atol = options.get("atol", 1e-5)
        self.rtol = options.get("rtol", 1e-5)

        # Parallelism
        self.parallel = options.get("parallel", 1)

        # Required by PET
        self.input_dict = options
        self.true_order = self.index
        self.all_data_types = self.datatype
        self.l_prim = [int(i) for i in range(len(self.report))]

    # ------------------------------------------------------------------

    def __call__(self, inputs: list | dict):
        """
        Run forward simulations for all ensemble members.

        Parameters
        ----------
        inputs : list of dict  or  dict
            One dict per ensemble member with keys ``x1``, ``x2``, ``mu``.

        Returns
        -------
        results : list
            One entry per ensemble member. Each entry is a list of dictionaries,
            one dictionary per report point.
        adjoints : list, optional
            One adjoint DataFrame per ensemble member when
            ``compute_adjoints=True``.
        """
        if isinstance(inputs, dict):
            inputs = [inputs]

        t_eval = np.asarray(self.report, dtype=float)
        # Prepend t=0 if absent so the solver has a valid starting point
        if t_eval[0] != 0.0:
            t_eval_full = np.concatenate([[0.0], t_eval])
            obs_mask = slice(1, None)
        else:
            t_eval_full = t_eval
            obs_mask = slice(None)

        args_list = [
            (member, idn, t_eval_full, self.datatype,
             self.compute_adjoints, self.atol, self.rtol)
            for idn, member in enumerate(inputs)
        ]

        if self.parallel > 1:
            with Pool(processes=self.parallel) as pool:
                raw = pool.map(_run_single, args_list)
        else:
            raw = [_run_single(a) for a in args_list]

        # Separate results / adjoints and trim t=0 padding
        if self.compute_adjoints:
            results, adjoints = [], []
            for res, sens in raw:
                res_trimmed = [
                    {k: np.array([v], dtype=float) for k, v in d.items()}
                    for d in res[obs_mask]
                ]
                results.append(res_trimmed)

                adj_rows = []
                for i in range(len(self.report)):
                    row = {}
                    for key in self.datatype:
                        row[key] = np.asarray(sens[key][i + (0 if t_eval[0] == 0.0 else 1)], dtype=float)
                    adj_rows.append(row)
                adj_df = pd.DataFrame(adj_rows, index=self.report)
                adj_df.index.name = self.report_type
                adjoints.append(adj_df)
            return results, adjoints

        return [
            [{k: np.array([v], dtype=float) for k, v in d.items()} for d in item[obs_mask]]
            for item in raw
        ]

    # ------------------------------------------------------------------

    def setup_fwd_run(self, **kwargs):
        """PET compatibility hook (no setup required for this simulator)."""
        return None

    # ------------------------------------------------------------------

    def run_fwd_sim(self, state: dict, member_i: int = 0, del_folder: bool = True):
        """
        Run the forward simulation for a single ensemble member.

        Mirrors the ``run_fwd_sim`` signature used in the other
        SimulatorWrap classes.

        Parameters
        ----------
        state : dict
            Keys: ``x1``, ``x2``, ``mu``. Values can be scalars or arrays with
            one element.
        member_i : int
            Ensemble member index (unused internally, kept for API
            compatibility).
        del_folder : bool
            Kept for PET compatibility. Unused.

        Returns
        -------
        result : list[dict]
            One dictionary per report point with keys equal to datatypes and
            values as 1D arrays.
        adj_df : pandas.DataFrame, optional
            Adjoint matrix in PET-compatible format; each cell contains a
            Jacobian row vector with derivatives w.r.t. [x1, x2, mu].
        """
        member_input = {
            "x1": float(np.asarray(state.get("x1", [1.0])).ravel()[0]),
            "x2": float(np.asarray(state.get("x2", [0.0])).ravel()[0]),
            "mu": float(np.asarray(state.get("mu", [1.0])).ravel()[0]),
        }

        t_eval = np.asarray(self.report, dtype=float)
        if t_eval[0] != 0.0:
            t_eval_full = np.concatenate([[0.0], t_eval])
            obs_mask = slice(1, None)
        else:
            t_eval_full = t_eval
            obs_mask = slice(None)

        args = (member_input, member_i, t_eval_full, self.datatype,
                self.compute_adjoints, self.atol, self.rtol)
        out = _run_single(args)

        if self.compute_adjoints:
            res, sens = out
            res = res[obs_mask]
            sens_offset = 0 if t_eval[0] == 0.0 else 1

            # Build PET output: list of dicts over report points
            pred = []
            for i in range(len(res)):
                row = {}
                for key in self.datatype:
                    row[key] = np.array([res[i][key]], dtype=float)
                pred.append(row)

            # Build adjoint dataframe: each cell holds d(y)/d[x1, x2, mu]
            # sens dict entries have keys datatype, values shape (n_obs, 3)
            adj_rows = []
            for i in range(len(self.report)):
                row = {}
                for key in self.datatype:
                    row[key] = np.asarray(sens[key][i + sens_offset], dtype=float)
                adj_rows.append(row)

            adj_df = pd.DataFrame(adj_rows, index=self.report)
            adj_df.index.name = self.report_type
            return pred, adj_df

        # No adjoints
        res = out[obs_mask]
        pred = []
        for i in range(len(res)):
            row = {}
            for key in self.datatype:
                row[key] = np.array([res[i][key]], dtype=float)
            pred.append(row)
        return pred
