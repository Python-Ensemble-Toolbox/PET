"""Quality assurance of the forecast (QA) and of the analysis (QC).

Four diagnostics, driven by the scheme through its hooks: after the prior
forecast and after every accepted iteration.

``calc_coverage``
    Is every observation inside the range the ensemble forecasts? Plots the
    forecast spread with the observations, marking those outside it, and logs
    how many fall outside per data type. Seismic (vector) data get the
    importance-scaled 2-D coverage maps of E. O. Lie (GeoCore).
``calc_mahalanobis``
    The model-deficiency diagnostic of Oliver (2020), *Diagnosing reservoir
    model deficiency for model improvement*: Mahalanobis distances between the
    observations and the perturbed forecast, singly (level 1) or in pairs and
    triples, logged as a ranked list with cross-plots of the worst.
``calc_kg``
    The ES-style Kalman gain each data type would apply to each parameter,
    ranked by size, so conflicting or dominant data can be spotted; field
    parameters can be written to the grid through the simulator.
``calc_da_stat``
    How far the parameters moved from the prior, in units of the prior
    standard deviation, per parameter group.

Data enters as the ensemble's frames -- observations, variances and
predictions indexed by report point with one column per data type, each cell
an array (``(1,)`` for point data, ``(n,)`` for vector data such as seismic)
or ``None`` -- and is adapted once, per data type, into the arrays the
diagnostics consume. Outputs go to a ``QAQC`` folder under the run's save
folder. Multilevel ensembles are not supported.

Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
"""

import logging
from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from scipy.io import loadmat

import pipt.misc_tools.analysis_tools as at

__all__ = ["QAQC"]

#: Data types treated as seismic (vector) data by the coverage maps.
SEISMIC_TYPES = ("bulkimp", "sim2seis", "avo", "grav")


def _finite_array(cell):
    """The cell as a flat float array, or ``None`` if it holds no usable value."""
    if cell is None:
        return None
    try:
        values = np.asarray(cell, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return values


def _rgb_to_hls(rgb):
    """Vectorised colorsys.rgb_to_hls on an (..., 3) array in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc, minc = rgb.max(axis=-1), rgb.min(axis=-1)
    lum = (maxc + minc) / 2
    delta = maxc - minc
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(delta == 0, 0.0,
                       np.where(lum <= 0.5, delta / (maxc + minc), delta / (2 - maxc - minc)))
        rc, gc, bc = (maxc - r) / delta, (maxc - g) / delta, (maxc - b) / delta
    hue = np.where(r == maxc, bc - gc, np.where(g == maxc, 2 + rc - bc, 4 + gc - rc))
    hue = np.where(delta == 0, 0.0, (hue / 6) % 1)
    return np.stack((hue, lum, sat), axis=-1)


def _hls_to_rgb(hls):
    """Vectorised colorsys.hls_to_rgb on an (..., 3) array in [0, 1]."""
    h, lum, s = hls[..., 0], hls[..., 1], hls[..., 2]
    m2 = np.where(lum <= 0.5, lum * (1 + s), lum + s - lum * s)
    m1 = 2 * lum - m2

    def channel(hue):
        hue = hue % 1
        return np.where(hue < 1 / 6, m1 + (m2 - m1) * hue * 6,
               np.where(hue < 0.5, m2,
               np.where(hue < 2 / 3, m1 + (m2 - m1) * (2 / 3 - hue) * 6, m1)))

    rgb = np.stack((channel(h + 1 / 3), channel(h), channel(h - 1 / 3)), axis=-1)
    return np.where(s[..., None] == 0, lum[..., None], rgb)


class QAQC:
    """Quality assurance of the forecast (QA) and the analysis (QC); see the module docstring.

    Parameters
    ----------
    keys : dict
        The ``dataassim`` config merged with the simulator's ``input_dict``.
        Read: ``assimindex`` (which report points are assimilated), and
        optionally ``actnum`` (path to an ``.npz`` with an ``actnum`` mask)
        and ``scale`` (a divisor applied to seismic data before plotting).
    data_df, data_var_df : PETDataFrame
        Observations and their variances, indexed by report point, one column
        per data type.
    logger : object, optional
        Anything with an ``info`` method. Defaults to ``logging.getLogger``.
    prior_info : dict, optional
        Per-parameter prior description (``nx``, ``ny``, ``nz``); needed by
        ``calc_kg`` and by grid output.
    sim : object, optional
        Simulator; used only for an optional ``write_to_grid`` method.
    ini_state : dict, optional
        The prior state, ``{parameter: (n, ne) array}``, as ``state_layout.to_dict(enX)``
        returns it; defines the parameter groups and the ensemble size.
    localization : object, optional
        The scheme's localization. Only the auto-adaptive kind is used, by
        ``calc_kg``; anything else is ignored.
    folder : str or Path, optional
        Where plots and grid files go. Default ``QAQC`` in the working directory.
    """

    def __init__(self, keys, data_df, data_var_df, logger=None, prior_info=None, sim=None,
                 ini_state=None, localization=None, folder="QAQC"):
        if "multilevel" in keys:
            raise NotImplementedError(
                "QA/QC is not available for multilevel ensembles: the diagnostics "
                "assume one prediction ensemble per report point."
            )
        self.keys = keys
        self.logger = logger if logger is not None else logging.getLogger("QAQC")
        self.prior_info = prior_info
        self.sim = sim
        self.ini_state = ini_state
        self.localization = localization if getattr(localization, "name", None) == "autoadaloc" else None
        self.list_state = list(ini_state.keys()) if ini_state else []
        self.ne = next(iter(ini_state.values())).shape[1] if ini_state else None
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.actnum = self._load_actnum(keys)

        self.data_types = list(data_df.columns)
        self._labels = list(data_df.index)
        self.l_prim = self._assimilated_positions(keys, len(self._labels))

        # Point data (one value per report point): (n_t, 1) arrays and the
        # positions they came from. Vector data (n values per report point,
        # e.g. seismic): concatenated over report points, plus the raw cells
        # for the per-vintage coverage maps.
        self.en_obs, self.en_var, self.en_time = {}, {}, {}
        self.en_obs_vec, self.en_var_vec, self.en_time_vec = {}, {}, {}
        self._obs_vector_cells = {}
        for typ in self.data_types:
            self._collect_observations(typ, data_df, data_var_df)

        # Filled by set().
        self.pred_data = None
        self.state = None
        self.lam = None
        self.en_fcst, self.en_fcst_vec, self._fcst_vector_cells = {}, {}, {}

    # ------------------------------------------------------------------
    # Adapting the frames
    # ------------------------------------------------------------------
    @staticmethod
    def _assimilated_positions(keys, n_points):
        """Positions (into the report-point index) of the assimilated data.

        ``assimindex`` is a list, or a list of lists for schemes that
        assimilate in several steps; every listed position counts here.
        """
        assim = keys.get("assimindex")
        if assim is None:
            return list(range(n_points))
        if not isinstance(assim, (list, tuple)):
            return [int(assim)]
        flat = []
        for item in assim:
            flat.extend(item if isinstance(item, (list, tuple)) else [item])
        return [int(x) for x in flat]

    @staticmethod
    def _load_actnum(keys):
        path = keys.get("actnum")
        if not path:
            return None
        try:
            return np.load(path)["actnum"].astype(bool)
        except Exception:
            return None

    def _collect_observations(self, typ, data_df, data_var_df):
        point, vector = [], []
        for pos in self.l_prim:
            label = self._labels[pos]
            obs = _finite_array(data_df.loc[label, typ])
            if obs is None:
                continue
            var = _finite_array(data_var_df.loc[label, typ]) if typ in data_var_df.columns else None
            if var is None or var.size not in (1, obs.size):
                self.logger.info(f"QAQC: no variance for {typ} at report point {label}; skipping it")
                continue
            var = np.broadcast_to(var, obs.shape)
            (point if obs.size == 1 else vector).append((pos, obs, var))

        self.en_obs[typ] = np.array([o for _, o, _ in point], dtype=float).reshape(-1, 1)
        self.en_var[typ] = np.array([v for _, _, v in point], dtype=float).reshape(-1, 1)
        self.en_time[typ] = [pos for pos, _, _ in point]
        if vector:
            self.en_obs_vec[typ] = np.concatenate([o for _, o, _ in vector])[:, None]
            self.en_var_vec[typ] = np.concatenate([v for _, _, v in vector])[:, None]
            self.en_time_vec[typ] = [pos for pos, _, _ in vector]
            self._obs_vector_cells[typ] = vector

    def set(self, pred_data, state=None, lam=None):
        """Hand over the current predictions, state and damping parameter.

        Parameters
        ----------
        pred_data : PETDataFrame
            Predictions aligned with the observation frame; each cell an array
            whose last axis is the ensemble.
        state : dict, optional
            Current state, ``{parameter: (n, ne) array}``.
        lam : float, optional
            The scheme's damping parameter (0 for schemes without one).
        """
        self.pred_data = pred_data
        self.state = state
        self.lam = lam
        for typ in self.data_types:
            rows = [np.asarray(pred_data.loc[self._labels[pos], typ], dtype=float).ravel()
                    for pos in self.en_time[typ]]
            self.en_fcst[typ] = (np.array(rows, dtype=float) if rows
                                 else np.empty((0, self.ne or 0)))
            cells = [np.asarray(pred_data.loc[self._labels[pos], typ], dtype=float)
                     for pos in self.en_time_vec.get(typ, [])]
            if cells:
                self._fcst_vector_cells[typ] = cells
                self.en_fcst_vec[typ] = np.concatenate(cells, axis=0)

    def _lumped(self, typ):
        """Point and vector data of one type stacked: forecast (nd, ne), observations and variances (nd, 1)."""
        parts = [(self.en_fcst.get(typ), self.en_obs.get(typ), self.en_var.get(typ)),
                 (self.en_fcst_vec.get(typ), self.en_obs_vec.get(typ), self.en_var_vec.get(typ))]
        parts = [(f, o, v) for f, o, v in parts if f is not None and f.size]
        if not parts:
            return None, None, None
        return tuple(np.concatenate(block, axis=0) for block in zip(*parts))

    def _save_figure(self, name):
        plt.savefig(self.folder / f"{name}.png", bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    def calc_coverage(self, line=None, field_dim=None, uxl=None, uil=None, contours=None,
                      uxl_c=None, uil_c=None):
        """Check whether the observations lie inside the ensemble's forecast range.

        For each point data type: a plot of the forecast ensemble over the
        report points with the observations, red where an observation lies
        above or below every member, and a log line with the count. For the
        first seismic data type present: the importance-scaled 2-D coverage
        maps, per vintage.

        Parameters
        ----------
        line : int, optional
            Also plot the 1-D coverage along this line of the seismic maps.
        field_dim : tuple, optional
            Grid dimensions of the seismic maps when no mask file is present.
        uxl, uil : array-like, optional
            Easting and northing coordinates of the map edges; default from a
            ``seglines.mat`` in the working directory, else grid indices.
        contours, uxl_c, uil_c : array-like, optional
            A contour field and its coordinates to draw over the maps.
        """
        self._require("pred_data")
        for typ in self.data_types:
            if typ in SEISMIC_TYPES or not self.en_obs[typ].size:
                continue
            fcst, obs = self.en_fcst[typ], self.en_obs[typ]
            below = (obs < fcst).all(axis=1)          # observation under every member
            above = (obs > fcst).all(axis=1)          # observation over every member
            times = np.asarray(self.en_time[typ])
            outside = int(below.sum() + above.sum())
            self.logger.info(f"QAQC coverage {typ}: {outside} of {obs.size} observations outside the ensemble range")

            plt.figure()
            plt.plot(times, fcst, c="0.35")
            plt.plot(times, obs, "g*")
            plt.plot(times[above], obs[above], "r*")
            plt.plot(times[below], obs[below], "r*")
            plt.title(f"{typ}: forecast range and observations")
            self._save_figure(typ.replace(" ", "_"))

        seismic = [typ for typ in SEISMIC_TYPES if typ in self._obs_vector_cells]
        if seismic:
            self._seismic_coverage(seismic[0], line, field_dim, uxl, uil, contours, uxl_c, uil_c)

    def _seismic_scaling(self):
        scale = self.keys.get("scale")
        if isinstance(scale, (list, tuple)) and len(scale) > 1:
            return float(scale[1])
        if isinstance(scale, (int, float)):
            return float(scale)
        return 1.0

    def _seismic_coverage(self, typ, line, field_dim, uxl, uil, contours, uxl_c, uil_c):
        scaling = self._seismic_scaling()
        observed = [obs / scaling for _, obs, _ in self._obs_vector_cells[typ]]
        predicted = [cell / scaling for cell in self._fcst_vector_cells.get(typ, [])]
        if len(predicted) != len(observed):
            self.logger.info(f"QAQC coverage {typ}: predictions missing, skipping the seismic maps")
            return

        if uxl is None and uil is None:
            try:
                seglines = loadmat("seglines.mat")
                uxl, uil = seglines["uxl"].flatten(), seglines["uil"].flatten()
            except Exception:
                uxl = uil = None

        nl = 0.25
        knots = np.array([-1, -np.finfo(float).eps, 0, .5, 1, 1 + np.finfo(float).eps, 2])
        channels = [interp1d(knots, np.array(c)) for c in (
            [0.1, 0.3, 0.8, 1.0, 0.8, 0.7, 0.5],
            [0.1, 0.3, 0.9, 1.0, 0.9, 0.4, 0.2],
            [0.4, 0.6, 0.8, 1.0, 0.8, 0.4, 0.2],
        )]

        for vint, (d_obs, d_pred) in enumerate(zip(observed, predicted)):
            try:
                mask = loadmat("mask_20.mat")[f"mask_{vint + 1}"].astype(bool).transpose()
            except Exception:
                if field_dim is None:
                    self.logger.info("QAQC coverage: no mask_20.mat and no field_dim given; skipping the seismic maps")
                    return
                mask = np.ones(field_dim, dtype=bool)
            data_real_reg = np.zeros(mask.shape)
            data_real_reg[mask] = d_obs
            data_reg = np.zeros(mask.shape + (d_pred.shape[1],))
            data_reg[mask] = d_pred

            d_min = data_reg.min(axis=2)
            d_max = data_reg.max(axis=2) + nl
            sat = 2 * np.minimum((d_max + data_real_reg) / np.max(d_max + data_real_reg), 0.5)
            attr = np.clip((data_real_reg - d_min) / (d_max - d_min), -1, 2)
            rgb = np.dstack([f(attr) for f in channels])

            x_edges = uxl if uxl is not None else [0, mask.shape[0]]
            y_edges = uil if uil is not None else [0, mask.shape[1]]
            extent = (x_edges[0], x_edges[-1], y_edges[-1], y_edges[0])

            def draw(image, title, name):
                plt.figure()
                plt.imshow(image, extent=extent)
                if contours is not None and uil_c is not None and uxl_c is not None:
                    plt.contour(uxl_c, uil_c, contours[::-1, :], levels=1, colors="black")
                    plt.xlim(extent[0], extent[1])
                    plt.ylim(extent[2], extent[3])
                    plt.xlabel("Easting (km)")
                    plt.ylabel("Northing (km)")
                plt.title(f"{title} - epsilon={nl}")
                self._save_figure(f"{name}_vint_{vint}")

            draw(rgb, "Coverage - not scaled by Importance", "coverage")
            # Importance scaling: darken the lightness channel where the
            # ensemble spread is small relative to the signal.
            hls = _rgb_to_hls(np.clip(rgb, 0, 1))
            hls[..., 1] = np.minimum(hls[..., 1] / (np.abs(sat - nl) / (1 - nl) * 1.5), 1.0)
            draw(np.clip(_hls_to_rgb(hls), 0, 1), "Coverage - scaled by Importance", "coverage_importance")
            draw(sat[::-1, :], "Importance", "importance")

            if line is not None:
                self._coverage_line(int(line), vint, data_reg, data_real_reg, nl, channels, x_edges)

    def _coverage_line(self, line, vint, data_reg, data_real_reg, nl, channels, x_edges):
        d_ens = np.squeeze(data_reg[:, line, :])
        d_real = np.squeeze(data_real_reg[:, line])
        scale = max(d_real)
        d_min = d_ens.min(axis=1)
        d_max = d_ens.max(axis=1) + nl
        sat = (2 * np.minimum((d_max + d_real) / scale, 0.5) - nl) / (1 - nl)
        attr = np.clip((d_real - d_min) / (d_max - d_min), -1, 2)
        colours = ListedColormap(np.column_stack([f(3 * np.arange(256) / 255 - 1) for f in channels]))
        x = np.arange(x_edges[0], x_edges[-1], (x_edges[-1] - x_edges[0]) / data_real_reg.shape[0])
        outline = np.column_stack((np.concatenate((x, x[::-1])), np.concatenate((d_min, d_max[::-1]))))

        for scaled, name in ((False, "coverage_1d"), (True, "coverage_1d_importance")):
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.add_patch(pat.Polygon(outline, closed=False, edgecolor="k", facecolor=np.array([.7, .7, .7])))
            segments = np.concatenate([np.array([x, d_real]).T.reshape(-1, 1, 2)[:-1],
                                       np.array([x, d_real]).T.reshape(-1, 1, 2)[1:]], axis=1)
            coloured = mcoll.LineCollection(segments, array=attr, cmap=colours, norm=plt.Normalize(-1, 2), linewidth=3)
            ax.add_collection(coloured)
            if scaled:
                alpha = np.clip(1 - sat, 0.0, 1.0)
                for i in range(len(x)):
                    seg = mcoll.LineCollection([[(x[i], d_min[i]), (x[i], d_max[i])]], colors="white",
                                               alpha=float(alpha[i]), linewidth=3)
                    ax.add_collection(seg)
            plt.colorbar(coloured)
            plt.xlim(x[0], x[-1])
            plt.ylim(0, scale)
            plt.title(f"1D coverage plot {'' if scaled else 'not '}scaled by Importance")
            self._save_figure(f"{name}_vint_{vint}")

    # ------------------------------------------------------------------
    # Kalman gain
    # ------------------------------------------------------------------
    def calc_kg(self, options=None):
        """Rank the ES-style Kalman gain each data type would apply to each parameter.

        For every data type, the gain of the ensemble mean is computed in the
        subspace of the forecast anomalies with the scheme's damping
        parameter (the ES/LM-EnRML form), per parameter. The largest gains by
        maximum and by mean are logged, and optionally plotted or written to
        the grid through the simulator.

        Parameters
        ----------
        options : dict, optional
            ``num_store`` (10): how many gains to keep in the ranked lists.
            ``unique_time`` (False): one gain per report point instead of one
            per data type over all its report points.
            ``plot_all_kg`` (False): plot or write every field gain, not just
            the ranked ones.
            ``only_log`` (True): log only; no plots or grid files.
            ``auto_ada_loc`` (True): apply the scheme's auto-adaptive
            localization, when it has one, to field parameters.
            ``write_to_resinsight`` (False): pass a time index to the grid writer.
        """
        opts = {"num_store": 10, "unique_time": False, "plot_all_kg": False, "only_log": True,
                "auto_ada_loc": True, "write_to_resinsight": False, **(options or {})}
        self._require("prior_info", "lam", "state")
        localize = opts["auto_ada_loc"] and self.localization is not None
        ranked = {"mean": [], "max": []}

        for typ in self.data_types:
            if opts["unique_time"]:
                for param in self.list_state:
                    scalar_gains = []
                    for ind, time in enumerate(self.en_time[typ]):
                        fcst = self.en_fcst[typ][ind][None, :]
                        obs, var = self.en_obs[typ][ind], self.en_var[typ][ind]
                        gain = self._gain(param, fcst, obs[:, None], var, localize)
                        if gain is None:
                            continue
                        if gain.size == 1:
                            scalar_gains.append(gain.item())
                        else:
                            self._rank(ranked, gain, (typ, param, time), opts["num_store"])
                            if not opts["only_log"] and opts["plot_all_kg"]:
                                self._write_field(gain, param, f"Kg_{param}_{typ}_{time}", time, opts)
                    if scalar_gains:
                        plt.figure()
                        plt.plot(self.en_time[typ], scalar_gains)
                        plt.title(f"Kalman gain of {param} from {typ}")
                        self._save_figure(f"Kg_{param}_{typ.replace(' ', '_')}")
            else:
                fcst, obs, var = self._lumped(typ)
                if fcst is None:
                    continue
                for param in self.list_state:
                    if self.state[param].shape[0] == 1:
                        continue
                    gain = self._gain(param, fcst, obs, var, localize)
                    if gain is None:
                        continue
                    self._rank(ranked, gain, (typ, param, None), opts["num_store"])
                    if not opts["only_log"] and opts["plot_all_kg"]:
                        self._write_field(gain, param, f"Kg-lump_{param}_{typ}", None, opts)

        newline = "\n"
        for kind in ("mean", "max"):
            entries = newline.join(f"{key}: {value:.4g}" for value, key in ranked[kind])
            self.logger.info(f"Calculations complete. {len(ranked[kind])} largest Kg {kind} values are:{newline}{entries}")

        if not opts["only_log"] and not opts["plot_all_kg"]:
            for kind in ("mean", "max"):
                for _, (typ, param, time) in ranked[kind]:
                    if time is None:
                        fcst, obs, var = self._lumped(typ)
                    else:
                        ind = self.en_time[typ].index(time)
                        fcst = self.en_fcst[typ][ind][None, :]
                        obs, var = self.en_obs[typ][ind][:, None], self.en_var[typ][ind]
                    gain = self._gain(param, fcst, obs, var, localize)
                    if gain is not None:
                        suffix = "" if time is None else f"-{time}"
                        self._write_field(gain, param, f"Kg-{kind}{suffix}_{param}_{typ}", time, opts)

    def _projection(self, pert, var):
        """The (ne, nd) operator taking a data residual to ensemble weights.

        Subspace form of ``C_md (C_dd + (1 + lam) C_d)^-1``: a truncated SVD
        of the forecast anomalies, then an eigendecomposition of the data
        covariance projected onto it. ``None`` if the ensemble has collapsed.
        """
        U, S, _ = at.truncSVD(pert, energy=0.99)
        if S.size == 0 or not np.any(S):
            return None
        Sinv = 1.0 / S
        X0 = (self.ne - 1) * ((Sinv[:, None] * U.T) @ (var[:, None] * U)) * Sinv[None, :]
        Lamb, Z = np.linalg.eigh(X0)
        X1 = (U * Sinv[None, :]) @ Z                                   # (nd, nr)
        return (pert.T @ X1) / ((self.lam + 1) + Lamb)[None, :] @ X1.T  # (ne, nd)

    def _gain(self, param, fcst, obs, var, localize):
        """Gain of the ensemble mean of ``param`` from data with forecast ``fcst`` (nd, ne)."""
        ne = min(self.ne, fcst.shape[1])
        fcst = fcst[:, :ne]
        pert = fcst - fcst.mean(axis=1, keepdims=True)
        X2 = self._projection(pert, np.asarray(var, dtype=float).ravel())
        if X2 is None:
            return None
        residual = obs - fcst                                          # (nd, ne)
        state = self.state[param][:, :ne]
        if localize and state.shape[0] > 1:
            anomalies = state - state.mean(axis=1, keepdims=True)
            projected = X2 @ residual                                  # (ne, ne)
            taper = self.localization(X=anomalies, Y=projected, parameters=[param],
                                      prior_info=self.prior_info)
            return ((taper * anomalies) @ projected).mean(axis=1)
        return state @ (X2 @ residual.mean(axis=1))

    @staticmethod
    def _rank(ranked, gain, key, keep):
        for kind, value in (("max", float(np.abs(gain).max())), ("mean", float(abs(gain.mean())))):
            ranked[kind].append((value, key))
            ranked[kind].sort(key=lambda item: item[0], reverse=True)
            del ranked[kind][keep:]

    def _write_field(self, values, param, name, time, opts):
        """Write a per-cell field to the grid through the simulator, if it can."""
        writer = getattr(self.sim, "write_to_grid", None) or getattr(getattr(self.sim, "flow", None), "write_to_grid", None)
        if writer is None:
            self.logger.info(f"QAQC: no grid writer on the simulator; {name} not written")
            return
        info = self.prior_info[param]
        dim = (info["nx"], info["ny"], info["nz"])
        if self.actnum is not None and self.actnum.sum() == values.size:
            data = np.zeros(self.actnum.shape)
            data[self.actnum] = values
            field = np.ma.array(data=data, mask=~self.actnum)
        elif self.actnum is None:
            field = np.ma.array(data=values, mask=np.zeros(values.shape, dtype=bool))
        else:
            return  # a surface parameter on a 3-D grid; no writer for that yet
        input_time = (len(self.l_prim) if time is None else time) if opts.get("write_to_resinsight") else None
        writer(field, name.replace(" ", "_"), str(self.folder), dim, input_time)

    # ------------------------------------------------------------------
    # Mahalanobis distance
    # ------------------------------------------------------------------
    def calc_mahalanobis(self, combi_list=(1, None)):
        """Rank the Mahalanobis distance between observations and the perturbed forecast.

        After Oliver (2020). The forecast is perturbed with the observation
        error (a fixed seed, so repeated calls agree), then each observation
        is scored against it alone (level 1), in pairs (2) or triples (3).
        The largest scores are logged; level 1 also draws cross-plots of the
        worst pairs.

        Parameters
        ----------
        combi_list : tuple
            Pairs ``(level, combine)``. ``combine`` is ``None`` to score each
            observation, or a string containing ``'time'`` or ``'vector'`` to
            first project each data type's series onto its leading principal
            component and score the data types.
        """
        self._require("pred_data")
        rng = np.random.default_rng(50)
        for combo in range(0, len(combi_list), 2):
            level = combi_list[combo]
            combine = combi_list[combo + 1] if combo + 1 < len(combi_list) else None
            self.logger.info(f"Starting level {level} calculations of Mahalanobis distance")

            if combine is None:
                types = [typ for typ in self.data_types if self.en_fcst.get(typ) is not None and self.en_fcst[typ].size]
                if not types:
                    return
                fcst = np.concatenate([self.en_fcst[typ] for typ in types], axis=0)
                obs = np.concatenate([self.en_obs[typ] for typ in types], axis=0)
                var = np.concatenate([self.en_var[typ] for typ in types], axis=0)
                labels = [(typ, pos) for typ in types for pos in self.en_time[typ]]
                fcst_pert = fcst + np.sqrt(var) * rng.standard_normal(fcst.shape)
            elif "time" in combine or "vector" in combine:
                labels, rows, obs_rows = [], [], []
                for typ in self.data_types:
                    series = self.en_fcst.get(typ)
                    if series is None or not series.size:
                        continue
                    pert = series + np.sqrt(self.en_var[typ]) * rng.standard_normal(series.shape)
                    _, _, vt = np.linalg.svd((pert - pert.mean(axis=1, keepdims=True)).T, full_matrices=False)
                    leading = vt[:1, :]                                  # (1, n_t)
                    rows.append((leading @ pert).ravel())
                    obs_rows.append((leading @ self.en_obs[typ]).ravel())
                    labels.append(typ)
                if not rows:
                    return
                fcst_pert, obs = np.array(rows), np.array(obs_rows)
            else:
                self.logger.info(f"Unknown combination {combine!r}; skipping")
                continue

            if level == 1:
                scores = (obs[:, 0] - fcst_pert.mean(axis=1)) ** 2 / fcst_pert.var(axis=1)
                top = np.argsort(scores)[::-1][:10]
                self.logger.info("Calculations complete. Largest values are:\n" + "\n".join(
                    f" data: {labels[i]}    Score: {scores[i]:.4g}" for i in top))
                self._crossplots(top, fcst_pert, obs, labels, combine)
            elif level in (2, 3):
                self._joint_scores(level, fcst_pert, obs, labels)
            else:
                self.logger.info(f"Mahalanobis level {level} is not implemented")

    def _joint_scores(self, level, fcst_pert, obs, labels):
        """Mahalanobis distance of every pair (level 2) or triple (3) of data."""
        n = len(fcst_pert)
        ne = fcst_pert.shape[1]
        scores = {}
        combos = ([(i, j) for i in range(n) for j in range(i + 1, n)] if level == 2
                  else [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)])
        for idx in combos:
            X = fcst_pert[list(idx)]
            mean = X.mean(axis=1)
            diff = X - mean[:, None]
            cov = diff @ diff.T / (ne - 1)
            res = obs[list(idx), 0] - mean
            try:
                scores[idx] = float(res @ np.linalg.solve(cov, res)) / 2
            except np.linalg.LinAlgError:
                continue
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]
        self.logger.info(f"Calculations complete. Largest level-{level} values are:\n" + "\n".join(
            f" data: {tuple(labels[i] for i in idx)}    Score: {score:.4g}" for idx, score in top))

    def _crossplots(self, top, fcst_pert, obs, labels, combine):
        if len(top) < 2:
            return
        pairs = [(top[0], top[1])] if len(top) < 4 else [(top[3], top[2]), (top[3], top[0])]
        for a, b in pairs:
            plt.figure()
            plt.plot(fcst_pert[a], fcst_pert[b], ".b")
            plt.plot(obs[a], obs[b], ".r")
            plt.xlabel(str(labels[a]) + (" (proj)" if combine else ""))
            plt.ylabel(str(labels[b]) + (" (proj)" if combine else ""))
            self._save_figure("crossplot_" + f"{labels[a]}-{labels[b]}".replace(" ", "_").replace("'", "")
                              .replace("(", "").replace(")", "").replace(",", "_t"))

    # ------------------------------------------------------------------
    # Update statistics
    # ------------------------------------------------------------------
    def calc_da_stat(self, options=None):
        """Log how far each parameter group moved from the prior.

        Per group: the mean prior and current standard deviation, and the
        percentage of parameters whose mean moved by more than one, two and
        three prior standard deviations.

        Parameters
        ----------
        options : dict, optional
            ``write_to_file`` (False): also write a field of these flags
            (-3..3) to the grid through the simulator.
        """
        self._require("state")
        write = bool(options and options.get("write_to_file"))
        lines = ["Statistics for updated parameters. Initial and final std, and percent larger than 1,2,3 initial std:"]
        for key in self.list_state:
            initial, current = self.ini_state[key], self.state[key]
            std0 = initial.std(axis=1)
            moved = current.mean(axis=1) - initial.mean(axis=1)
            stds = (float(std0.mean()), float(current.std(axis=1).mean()))
            pct = tuple(float(100 * np.mean(np.abs(moved) > k * std0)) for k in (1, 2, 3))
            lines.append(f"Group {key}: std {stds[0]:.4g} -> {stds[1]:.4g}; "
                         f"{pct[0]:.1f}% / {pct[1]:.1f}% / {pct[2]:.1f}% beyond 1 / 2 / 3 std")
            if write and moved.size > 1:
                flags = np.zeros(moved.shape)
                for k in (1, 2, 3):
                    flags[moved > k * std0] = k
                    flags[moved < -k * std0] = -k
                self._write_field(flags, key, f"da_stat_{key}", None, {})
        self.logger.info("\n".join(lines))

    def _require(self, *names):
        """Raise if any of the named inputs is still unset (``set()`` provides all but ``prior_info``)."""
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            hint = "" if missing == ["prior_info"] else "; call set() first"
            raise ValueError(f"QAQC needs {', '.join(missing)}{hint}")
