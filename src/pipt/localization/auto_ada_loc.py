"""Adaptive localization implementation."""
import numpy as np
from misc.sampling import random_stream
from typing import Union
from scipy.special import expit
from pipt.localization.common import (
    LocalizationBase,
)

__all__ = ["AutoAdaptiveLocalization"]

class AutoAdaptiveLocalization(LocalizationBase):
    """Adaptive localization strategy and engine implementation."""

    name = "autoadaloc"

    def __init__(self, info: Union[dict, list], rng=None):
        """
        Initialize the AutoAdaptiveLocalization instance.

        All configuration is supplied through the ``info`` dictionary, which maps
        directly to a ``[dataassim.localization]`` table in a TOML config file.

        Parameters
        ----------
        info : dict or list
            Localization configuration. Recognised keys:

            **field** : list of int, *required*
                Grid dimensions. For a 3-D reservoir use ``[nz, nx, ny]``;
                for a 2-D field ``[nx, ny]`` is sufficient. Only the product
                (total cell count) is used by this class.

            **actnum** : str, *optional*
                Path to a ``.npz`` file whose first array is a boolean mask
                of active cells. When supplied, only active cells are counted
                toward ``default_num_active``. Default: ``None`` (all cells
                are considered active).

            **threshold** : {``"adaptive"``, ``"fixed"``, ``"universal"``}, *optional*
                Method used to compute the correlation threshold below which
                a correlation is deemed indistinguishable from sampling noise:

                - ``"adaptive"`` — threshold = ``cutoff * sigma``, where *sigma*
                  is estimated column-wise from shuffled correlations via the
                  MAD estimator. The ``cutoff`` parameter controls how many noise
                  standard deviations to use as the cut-off.
                - ``"fixed"`` — threshold equals ``cutoff`` directly; no noise
                  estimation is performed. Use when you want a deterministic,
                  reproducible cut-off independent of the ensemble.
                - ``"universal"`` — threshold = ``sqrt(2 * log(N)) * sigma``;
                  adapts automatically to ensemble size without requiring
                  ``cutoff`` to be tuned.

                Default: ``"adaptive"``.

            **cutoff** : float, *optional*
                Threshold value or noise multiplier (interpretation depends on
                ``threshold``). Larger values suppress more correlations.
                Default: ``0.3``.

            **type** : {``"hard"``, ``"soft"``, ``"sigm"``}, *optional*
                Tapering strategy applied once the threshold is known:

                - ``"hard"`` — binary mask: 1 where |r| ≥ threshold, 0
                  elsewhere. Sharp cut-off, computationally efficient.
                - ``"soft"`` — smooth rational-function taper that transitions
                  gradually around the threshold. Avoids discontinuities in
                  the localization operator.
                - ``"sigm"`` — sigmoid-based taper; similar smoothness to
                  ``"soft"`` but with a different shape near the transition.

                Default: ``"hard"``.

        Examples
        --------
        Minimal TOML block inside ``[dataassim]`` using fixed thresholding:

        ```toml
        [dataassim.localization]
        name       = "autoadaloc"
        field      = [1, 20, 20]   # [nz, nx, ny]
        threshold  = "fixed"
        cutoff     = 0.4
        type       = "hard"
        ```

        Noise-adaptive thresholding with a smooth taper:

        ```toml
        [dataassim.localization]
        name       = "autoadaloc"
        field      = [2, 30, 40]   # two-layer, 30×40 lateral grid
        actnum     = "active_cells.npz"
        threshold  = "universal"   # adapts to ensemble size automatically
        type       = "soft"
        ```

        Large state vector — skip forming the full cross-covariance:

        ```toml
        [dataassim.localization]
        name       = "autoadaloc"
        field      = [5, 100, 100]
        threshold  = "fixed"
        cutoff     = 0.3
        type       = "hard"
        ```
        """
        # The stream the shuffle below draws from; the global one unless the run is seeded.
        self.rng = rng if rng is not None else random_stream()
        self.field, self.actnum = self.config_common(info)
        self.cutoff = self._cutoff_from(info)
        self.threshold  = info.get("threshold", "adaptive")
        self.tapertype  = info.get("type", "hard")
        self.parameters = info.get("parameters", ['NA'])
        self.projection = info.get("projection", "rank-r")

        # Ensure that the tapering type is valid
        if self.tapertype not in ["hard", "soft", "sigm"]:
            raise ValueError(
                f"Invalid tapering type '{self.tapertype}'. "
                "Supported types are 'hard', 'soft', and 'sigm'."
            )

        # Ensure that the threshold method is valid
        if self.threshold not in ["adaptive", "fixed", "universal"]:
            raise ValueError(
                f"Invalid threshold method '{self.threshold}'. "
                "Supported methods are 'adaptive', 'fixed', and 'universal'."
            )

    def __call__(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            parameters: list[str]=None,
            prior_info: dict=None
        ) -> np.ndarray:
        """
        Calculate truncated cross-covariance matrix.

        Parameters
        ----------
        X : ndarray, shape (nx, ne)
            State perturbation ensemble.

        Y : ndarray, shape (ny, ne)
            Projected predicted data ensemble.

        parameters : list[str]
            Ordered list of parameters corresponding to blocks in X.

        prior_info : dict, optional
            Prior information for each parameter. If provided,
            ``prior_info[param]["active"]`` specifies the number of
            active variables associated with the parameter.

        Returns
        -------
        ndarray, shape (nx, ny)
            Tapered matrix containing the tapering coefficients for the cross-covariance between X and Y.
        """
        parameters = self.parameters if parameters is None else parameters
        prior_info = {} if prior_info is None else prior_info

        corr = self.corr_matrix(X, Y) # Shape: (nx, ny)
        corr_shuffled = self.corr_matrix(
            X[:, self.rng.permutation(X.shape[1])],
            Y,
        )

        default_num_active = (
            np.sum(self.actnum) if (self.actnum is not None) else np.prod(self.field)
        )

        taper = np.ones_like(corr)
        row_start = 0
        for param in parameters:

            if param == "NA":
                num_active = taper.shape[0] - row_start
            else:
                param_info = prior_info.get(param, {})
                num_active = int(param_info.get("active", default_num_active))

            rows = slice(row_start, row_start + num_active)
            taper[rows] = self.tapering_function(
                corr[rows],
                corr_shuffled[rows],
            )
            row_start += num_active

        return taper


    @staticmethod
    def _cutoff_from(info: dict) -> float:
        """How many noise standard deviations a correlation must clear to survive.

        This is what used to be called ``nstd``, and it was carried as the value of the
        ``autoadaloc`` key itself -- ``AUTOADALOC 2`` meant two. Reading only ``cutoff``
        left such a config running at the default while the number the user wrote was
        ignored, which changes the taper and so the posterior without any error. All
        three spellings are accepted; ``autoadaloc`` is also set to ``True`` as a plain
        mode flag, which is not a value and is skipped.
        """
        for key in ("cutoff", "nstd", "autoadaloc"):
            value = info.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.3

    def tapering_function(self, corr_values: np.ndarray, corr_values_shuffled: np.ndarray) -> np.ndarray:

        """
        Compute tapering coefficients from sample correlations.

        The tapering coefficients are used to suppress correlations that are
        indistinguishable from noise. A noise level is estimated for each
        observation variable from the corresponding shuffled correlations using
        the median absolute deviation (MAD),

            sigma = median(|r_shuffled|) / 0.6745

        which provides a robust estimate of the standard deviation under the
        assumption of Gaussian noise.

        Depending on the localization settings, the correlation threshold is
        computed using one of the following methods:

        - ``"adaptive"`` (default):
            threshold = cutoff * sigma
        - ``"fixed"``:
            threshold = cutoff
        - ``"universal"``:
            threshold = sqrt(2 log(N)) * sigma

        Tapering can then be applied using one of three strategies:

        - ``"hard"`` (default):
            correlations above the threshold are assigned a taper value of 1,
            otherwise 0.
        - ``"soft"``:
            smooth tapering based on ``rational_function``.
        - ``"sigm"``:
            sigmoid-based tapering using ``rational_function_sigmoid``.

        Parameters
        ----------
        corr_values : ndarray of shape (nx, ny)
            Sample correlation matrix.

        corr_values_shuffled : ndarray of shape (nx, ny)
            Correlation matrix computed from shuffled or randomized ensembles.
            Used to estimate the noise level of the correlations.

        Returns
        -------
        ndarray of shape (nx, ny)
            Tapering coefficients in the interval [0, 1]. These coefficients
            can be applied element-wise to the correlation matrix to reduce
            the influence of correlations attributed to sampling noise.
        """
        taper_coeff = np.zeros_like(corr_values)
        for i in range(corr_values.shape[1]):
            corr = corr_values[:, i]

            # Estimate noise level from shuffled correlations:
            mad_to_std = 1 / 0.6745
            noise_std  = np.median(np.abs(corr_values_shuffled[:, i])) * mad_to_std

            # Compute threshold
            if self.threshold == "fixed":
                threshold = self.cutoff
            elif self.threshold == "universal":
                threshold = np.sqrt(2 * np.log(corr.size)) * noise_std
            else:  # "adaptive"
                threshold = self.cutoff * noise_std

            # Compute taper coefficients
            if self.tapertype == "soft":
                taper = self.rational_function(
                    1 - np.abs(corr),
                    1 - threshold,
                )
            elif self.tapertype == "sigm":
                taper = self.rational_function_sigmoid(
                    np.abs(corr),
                    threshold,
                )
            else:
                taper = np.zeros_like(corr)
                taper[np.abs(corr) > threshold] = 1.0

            taper_coeff[:, i] = taper

        return taper_coeff


    def rational_function(self, distance, length_scale):
        """Piecewise rational taper of ``distance`` at ``length_scale``: 1 inside the scale, decaying to 0 at twice the scale."""
        z_ratio = np.absolute(distance) / length_scale
        idx_inner = np.where(z_ratio <= 1)
        idx_outer = np.where(z_ratio <= 2)
        idx_transition = np.setdiff1d(idx_outer, idx_inner)

        taper = np.zeros(len(z_ratio))

        taper[idx_inner] = (
            1
            - (np.power(z_ratio[idx_inner], 5) / 4)
            + (np.power(z_ratio[idx_inner], 4) / 2)
            + (5 * np.power(z_ratio[idx_inner], 3) / 8)
            - (5 * np.power(z_ratio[idx_inner], 2) / 3)
        )

        taper[idx_transition] = (
            (np.power(z_ratio[idx_transition], 5) / 12)
            - (np.power(z_ratio[idx_transition], 4) / 2)
            + (5 * np.power(z_ratio[idx_transition], 3) / 8)
            + (5 * np.power(z_ratio[idx_transition], 2) / 3)
            - 5 * z_ratio[idx_transition]
            - np.divide(2, 3 * z_ratio[idx_transition])
            + 4
        )

        return taper

    @staticmethod
    def rational_function_sigmoid(distance, length_scale):
        """A steep sigmoid taper switching at ``length_scale``."""
        steepness = 50
        return expit((distance - length_scale) * steepness)

    @staticmethod
    def corr_matrix(X, Y, eps=1e-6):
        """
        Compute the correlation matrix between two ensemble matrices X and Y.

        Parameters
        ----------
        X : np.ndarray, shape (nx, ne)
        Y : np.ndarray, shape (ny, ne)
        eps : float, optional, default=1e-6
            Small value to avoid division by zero when computing standard deviations.

        Returns
        -------
        corr : np.ndarray, shape (nx, ny)
            The correlation matrix between X and Y.
        """
        stdX = np.std(X, axis=1)
        stdY = np.std(Y, axis=1)

        nx = X.shape[0]
        corr = np.corrcoef(X, Y)[:nx, nx:]
        corr[stdX < eps, :] = 0
        corr[:, stdY < eps] = 0

        return np.nan_to_num(corr)


