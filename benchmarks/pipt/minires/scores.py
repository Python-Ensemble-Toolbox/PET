"""What an ensemble is scored on: the data it was shown, the unknowns, and the forecast it was not shown.

The misfit is what an assimilation minimises, so it says least about it: an ensemble can be
made to fit its history while moving away from the truth. Hence the other scores, which a
twin experiment can compute because it knows the truth.
"""

import numpy as np

from cases import DATATYPES, FORECAST_STEPS, OBS_STEPS, SCALARS, Twin

WWCT = [i for i, d in enumerate(DATATYPES) if d.startswith("WWCT")]


def crps(ensemble: np.ndarray, truth) -> np.ndarray:
    """The continuous ranked probability score of each row of ``ensemble`` (members along axis 1).

    ``E|X - y| - E|X - X'| / 2``, the latter by sorting: the mean absolute difference of the
    members is ``2/ne^2 sum_i (2i - ne - 1) x_(i)``. In the units of the variable; zero is perfect.
    """
    ne = ensemble.shape[1]
    x = np.sort(ensemble, axis=1)
    spread = 2 / ne**2 * (x @ (2 * np.arange(1, ne + 1) - ne - 1))
    return np.mean(np.abs(ensemble - np.atleast_1d(truth)[:, None]), axis=1) - spread / 2


def scores(states: dict, responses: np.ndarray, twin: Twin) -> dict:
    """Score one ensemble.

    Parameters
    ----------
    states : dict
        ``{variable: (size, ne)}``, the ensemble's unknowns.
    responses : ndarray
        ``(ne, NSTEPS, len(DATATYPES))``, each member simulated to the end.
    twin : Twin
        The truth and its observations.
    """
    out = {}

    # The history: mean squared normalized misfit, over members and observations
    history = responses[:, np.array(OBS_STEPS) - 1]
    out["misfit"] = float(np.nanmean(((history - twin.obs) / twin.std) ** 2))

    # The forecast: RMS error of the ensemble mean's water cut, over the unseen steps
    forecast = np.nanmean(responses[:, np.array(FORECAST_STEPS) - 1][..., WWCT], axis=0)
    truth = twin.response[np.array(FORECAST_STEPS) - 1][:, WWCT]
    out["forecast_wwct_rmse"] = float(np.sqrt(np.mean((forecast - truth) ** 2)))
    members = responses[:, np.array(FORECAST_STEPS) - 1][..., WWCT].reshape(len(responses), -1).T
    out["forecast_wwct_crps"] = float(np.nanmean(crps(members, truth.ravel())))

    # The field: error and correlation of the ensemble mean, and the spread left
    logk, true_logk = states["permx"], twin.truth["permx"]
    mean = logk.mean(axis=1)
    out["logk_rmse"] = float(np.sqrt(np.mean((mean - true_logk) ** 2)))
    out["logk_corr"] = float(np.corrcoef(mean, true_logk)[0, 1])
    out["logk_spread"] = float(logk.std(axis=1, ddof=1).mean())
    out["logk_crps"] = float(crps(logk, true_logk).mean())

    # The scalars: error of the ensemble mean, and the spread left
    for name in SCALARS:
        values = states[name][0]
        out[f"{name}_error"] = float(abs(values.mean() - twin.truth[name][0]))
        out[f"{name}_spread"] = float(values.std(ddof=1))
        out[f"{name}_crps"] = float(crps(values[None], twin.truth[name][0])[0])
    return out
