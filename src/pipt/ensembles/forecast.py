"""Forecast support for assimilation ensembles.

Running the forward simulator and turning its raw output into ``pred_data`` is
ensemble work, not loop work: it reads ``sim``, ``enX``, ``data_df`` and the
compression machinery, and it writes ``pred_data``. It lived on
``pipt.loop.assimilation.Assimilate`` only because that class historically drove
every iteration.

:class:`AssimilationScheme` expects its ensemble collaborator to expose a
public :meth:`ForecastMixin.forecast`, so the forecast lives here and the loop
delegates to it. Mixed into :class:`pipt.ensembles.AssimilationEnsemble`.
"""

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from misc.structures import PredictedData

import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract

__all__ = ["ForecastMixin", "OutlierMixin"]


class ForecastMixin:
    """Forward simulation and predicted-data preparation."""

    RESTART_RESULTS_FILE = "restart_sim_results.pkl"
    SIM_RESULTS_FILE = "sim_results.pkl"

    def forecast(self, enX) -> None:
        """Run forecast simulations and prepare predicted data for analysis.

        Parameters
        ----------
        enX
            The state to predict on. Passed in rather than read off the
            ensemble, so a scheme can forecast a *trial* state without first
            parking it somewhere for this method to find.
        """
        if self._load_restart_prediction_if_available():
            return

        self.calc_prediction(enX)
        self.pred_data = self._predicted_data()
        self.adjoints = self._adjoint_array()

        # Multilevel runs correct each level towards the reference level's mean.
        if getattr(self, "multilevel", None) is not None:
            self.treat_modeling_error()

        self._apply_prediction_scaling()
        self._save_reconstructed_forecast_if_requested()
        self._save_forecast_debug()

    def _predicted_data(self):
        """The forecast as the analyses see it: the layout's rows, filled from each member's output.

        One container per level for a multilevel ensemble. Scaling follows the
        observations: when ``data_df`` was max-min scaled, so are these, with
        the same minimum and maximum per data type. Compressed data types are
        reduced to the observed vintage's wavelet coefficients on the way in.
        """
        scale = (self.data_df.scale_min, self.data_df.scale_max) if self.data_df.is_scaled else None
        position = self._record_positions()
        transform = self._row_transform()
        levels = [PredictedData.from_members(self.data_layout, members, position=position, scale=scale, transform=transform)
                  for members in self.member_outputs]
        return levels if getattr(self, "multilevel", None) is not None else levels[0]

    # ------------------------------------------------------------------
    # Wavelet compression of seismic data types (the `compress` option)
    # ------------------------------------------------------------------
    def _compressed_rows(self) -> dict:
        """``{(label, datatype): vintage}`` for the observed cells the reader compressed, in its order.

        The reader walks the observed frame label-major then type and numbers
        the compressed cells it meets; the layout walks the same way, so the
        n-th compressed row is vintage n. Cells beyond the masks given stay
        uncompressed on both sides.
        """
        if not self.sparse_info or not self.sparse_data:
            return {}
        types = self.sparse_info["compress_data"]
        types = [types] if isinstance(types, str) else list(types)
        rows = [row for row in self.data_layout.rows if row.datatype in types]
        return {(row.label, row.datatype): vintage for vintage, row in enumerate(rows[:len(self.sparse_data)])}

    def _sim2seis_scale(self):
        """The `sim2seis` scaling factor from ``scale_results.pkl``, read once; ``None`` when not in use."""
        if not extract.is_enabled(self.keys_da.get("post_process_forecast", False)):
            return None
        if self.scale_val is None and os.path.exists("scale_results.pkl"):
            with open("scale_results.pkl", "rb") as file:
                scale = pickle.load(file)
            self.scale_val = np.sum(scale[0]) / len(scale[0])
        return self.scale_val

    def _row_transform(self):
        """What a member's raw values go through before entering the matrix; ``None`` when nothing does.

        Data types containing ``sim2seis`` are divided by the sim2seis scale
        when one is configured; compressed vintages become their leading
        wavelet coefficients, through the same :class:`SparseRepresentation`
        that reduced the observed vintage, so the leading indices match. The
        reconstruction of each compressed member is kept only when
        ``saveforecast`` will write it.
        """
        compressed = self._compressed_rows()
        scale_val = self._sim2seis_scale()
        if not compressed and scale_val is None:
            return None
        keep_reconstruction = compressed and "saveforecast" in self.sim.input_dict
        self.data_rec = [[] for _ in range(len(self.sparse_data))] if compressed else []

        def transform(row, values):
            if scale_val is not None and "sim2seis" in row.datatype:
                values = values / scale_val
            vintage = compressed.get((row.label, row.datatype))
            if vintage is not None:
                values, wdec_rec = self.sparse_data[vintage].compress(values)
                if keep_reconstruction:
                    self.data_rec[vintage].append(self.sparse_data[vintage].reconstruct(wdec_rec))
            return values

        return transform

    def _adjoint_array(self):
        """The members' adjoints as ``(nd, nx, ne)`` in layout order, scaled with the data; ``None`` without adjoints.

        Each member's adjoint is a frame whose cells hold the sensitivity of
        that cell's values to the ``nx`` state variables. Only observed cells
        are taken, so the array lines up with ``pred_data`` row for row.
        """
        members = self.member_adjoints
        if not members:
            return None
        cells = {row: [np.asarray(member.loc[row.label, row.datatype], dtype=float).reshape(row.size, -1)
                       for member in members] for row in self.data_layout.rows}
        nx = next(iter(cells.values()))[0].shape[1]
        out = np.empty((self.data_layout.nd, nx, len(members)))
        for row, blocks in cells.items():
            for j, block in enumerate(blocks):
                out[row.rows, :, j] = block
        if self.data_df.is_scaled:
            span = self.data_df.scale_max - self.data_df.scale_min
            for row in self.data_layout.rows:
                out[row.rows] = (out[row.rows] - 0) / span[row.datatype]
        return out

    def _record_positions(self):
        """Where each observed label sits in a member's records: the simulator's ``true_order``, else the label itself."""
        order = getattr(self.sim, "true_order", None)
        if order is None:
            return None
        positions = pd.Index(order[1]).get_indexer(list(self.data_layout.labels))
        missing = [label for label, pos in zip(self.data_layout.labels, positions) if pos < 0]
        if missing:
            raise ValueError(f"the simulator reports no values at observed labels {missing!r}")
        return dict(zip(self.data_layout.labels, positions))

    def treat_modeling_error(self) -> None:
        """Shift every coarser level so each row's ensemble mean matches the finest level's."""
        reference = self.pred_data[-1].matrix.mean(axis=1)
        for level in self.pred_data[:-1]:
            level.matrix += (reference - level.matrix.mean(axis=1))[:, None]

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------
    @property
    def _saving_enabled(self) -> bool:
        return "nosave" not in self.keys_da

    @property
    def save_folder(self) -> str | None:
        """Folder for run artifacts, or ``None`` when saving is disabled.

        Both ``savefolder`` and ``save_folder`` are accepted, as POPT's
        optimizers do -- only the former used to be read, so a config written
        with the underscored spelling silently wrote to ``Results`` instead.
        Reading this creates nothing; :meth:`_save_path` makes the folder when
        something is about to be written into it.
        """
        if not self._saving_enabled:
            return None
        return self.keys_da.get("savefolder", self.keys_da.get("save_folder", "Results"))

    def _save_path(self, filename: str) -> str:
        """Path of ``filename`` inside the save folder, which is created here."""
        if self.save_folder is None:
            raise RuntimeError("Cannot save results because saving is disabled.")
        os.makedirs(self.save_folder, exist_ok=True)
        return os.path.join(self.save_folder, filename)

    # ------------------------------------------------------------------
    # Forecast steps
    # ------------------------------------------------------------------
    def _load_restart_prediction_if_available(self) -> bool:
        # A hand-placed file: a saved forecast copied to this name in the
        # working directory supplies the forecast a crashed run had already
        # finished. It is honoured only on a restart, so a file left behind
        # cannot silently stand in for a fresh forecast on an ordinary run.
        if not self.restart or not os.path.exists(self.RESTART_RESULTS_FILE):
            return False

        with open(self.RESTART_RESULTS_FILE, "rb") as file:
            self.sim_data = pickle.load(file)

        self.pred_data = self._container_from_frame(self.sim_to_pred_data(self.sim_data))

        # Consumed once; it then lives with the other results under the name a
        # saved forecast gets (in the working directory when saving is off).
        used = self.SIM_RESULTS_FILE if self.save_folder is None else self._save_path(self.SIM_RESULTS_FILE)
        os.replace(self.RESTART_RESULTS_FILE, used)
        self.logger("--- Restart sim results used ---")
        return True

    def _apply_prediction_scaling(self) -> None:
        """Multiply the predictions of the data types named by ``scale`` by its factor."""
        if "scale" not in self.keys_da:
            return

        scale_keys, scale_factor = self.keys_da["scale"]
        if isinstance(scale_keys, str):
            scale_keys = [scale_keys]
        levels = self.pred_data if isinstance(self.pred_data, list) else [self.pred_data]
        for level in levels:
            for datatype in scale_keys:
                for rows in level.rows_of(datatype):
                    level.matrix[rows] *= scale_factor

    def _save_forecast_debug(self) -> None:
        if "saveforecast" not in self.sim.input_dict:
            return
        if not self._saving_enabled:
            return

        forecast = self.sim_data
        if self.data_df.is_scaled:
            forecast = forecast.copy().invert_scale()

        with open(self._save_path(self.SIM_RESULTS_FILE), "wb") as file:
            pickle.dump(forecast, file)

    def _container_from_frame(self, frame):
        """A ``PredictedData`` (one per level) from a prediction frame, for paths that still produce frames."""
        if isinstance(frame, list):
            return [PredictedData.from_frame(self.data_layout, level, self.ne) for level in frame]
        return PredictedData.from_frame(self.data_layout, frame, self.ne)

    def sim_to_pred_data(self, pred: Any) -> Any:
        '''
        Filter the simulator output to match the structure of the predicted data expected.

        Parameters
        ----------
        pred : Any
            The raw output from the simulator, which may be a list of DataFrames or a single DataFrame.

        Returns
        -------
        Any
            The processed predicted data, structured to match the ensemble's expected format for analysis.
        '''
        if isinstance(pred, list):
            return [self.sim_to_pred_data(frame) for frame in pred]
        index = self.data_df.index
        columns = self.data_df.columns
        return pred.filter_dataframe(index=index, columns=columns)

    def _save_reconstructed_forecast_if_requested(self) -> None:
        """Write the reconstructed compressed vintages, ``(n_raw, ne)`` per vintage, when ``saveforecast`` asks."""
        if "saveforecast" not in self.sim.input_dict or not self.data_rec:
            return
        self.data_rec = [np.asarray(members).T for members in self.data_rec]
        with open("rec_results.pkl", "wb") as file:
            pickle.dump(self.data_rec, file)


class OutlierMixin:
    """Replacement of outlier ensemble members.

    Ensemble work, like the forecast: it rewrites ``pred_data``, ``sim_data``
    and the state matrix in place. Called between forecast and scoring, so the
    replacement feeds into the misfit the scheme sees.
    """

    def remove_outliers(self, enX):
        """Replace outlier ensemble members with resampled non-outliers.

        Returns the state with outliers resampled -- the same object when
        there is nothing to replace. Returned rather than written back,
        because the caller owns the state being forecast.
        """
        outlier_idx, non_outlier_idx = at.get_outlier_index(
            self.pred_data.matrix, self.obs_vector, self.obs_variance,
        )
        if len(outlier_idx) == 0:
            return enX
        idx = np.arange(self.ne)
        for outlier in outlier_idx:
            new_idx = self.rng.choice(non_outlier_idx)
            idx[outlier] = new_idx
            self.logger(f"Replaced outlier {outlier} with member {new_idx}")

        self.pred_data = self.pred_data.take_members(idx)

        # The full forecast follows the members: reorder the raw outputs and
        # let the frame view be rebuilt when next asked for. A forecast loaded
        # from a file exists only as a frame; its cells with no data are None
        # and are left alone (na_action), instead of failing on `.ndim`.
        if getattr(self, "member_outputs", None):
            self.member_outputs = [[members[i] for i in idx] for members in self.member_outputs]
            self._sim_data = None
        elif getattr(self, "sim_data", None) is not None:
            def filter_outliers(cell):
                return cell[..., idx] if cell.ndim > 1 else cell[idx]
            self.sim_data = self.sim_data.map(filter_outliers, na_action='ignore')

        # The adjoint belongs to the member it was evaluated at, so it moves
        # with the state and the predictions -- a member whose gradient came
        # from a different member is not a member of anything.
        if getattr(self, "adjoints", None) is not None:
            self.adjoints = self.adjoints[..., idx]
            if getattr(self, "member_adjoints", None):
                self.member_adjoints = [self.member_adjoints[i] for i in idx]

        return enX[:, idx]
