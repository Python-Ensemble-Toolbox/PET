"""Gaussian control perturbations: ensemble estimates of the gradient, the Hessian and the sensitivity used by SmcOpt."""
# External imports
import numpy as np
import warnings

from copy import deepcopy

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.ensembles.ensemble_base import EnsembleOptimizationBase

__all__ = ['GaussianEnsemble']

class GaussianEnsemble(EnsembleOptimizationBase):
    """
    Gaussian Ensemble class for ensemble-based optimization.

    Methods
    -------
    gradient(x, *args, **kwargs)
        Ensemble gradient

    hessian(x, *args, **kwargs)
        Ensemble hessian

    calc_ensemble_weights(self,x, *args, **kwargs):
        Calculate weights used in sequential monte carlo optimization
    """

    def __init__(self, options, simulator, objective):
        """
        Parameters
        ----------
        options : dict
            Options for the ensemble class

            - disable_tqdm: supress tqdm progress bar for clean output in the notebook
            - ne: number of perturbations used to compute the gradient
            - state: name of state variables passed to the .mako file
            - prior_<name>: the prior information the state variables, including mean, variance and variable limits
            - num_models: number of models (if robust optimization) (default 1)
            - transform: transform variables to [0,1] if true (default true)
            - natural_gradient: use natural gradient if true (default false)

        simulator : callable
            The forward simulator (e.g. flow)

        objective : callable
            The objective function (e.g. npv)
        """

        # Initialize PETEnsemble
        super().__init__(options, simulator, objective)

        # Inflation factor used in SmcOpt
        self.inflation_factor = None
        self.survival_factor = None
        self.particles = []  # list in case of multilevel
        self.particle_values = []  # list in case of multilevel
        self.resample_index = None

    def gradient(self, x, *args, **kwargs):
        """
        Estimate the ensemble gradient (EnOpt) at a given state.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, ).
        args : tuple
            First positional argument must be the covariance matrix with shape
            (number of controls, number of controls).

        Returns
        -------
        ndarray
            Ensemble gradient, shape (number of controls, ).

        Raises
        ------
        ValueError
            If required inputs are missing or have invalid shapes.
        """
        if len(args) < 1:
            raise ValueError("gradient requires covariance matrix as first positional argument.")

        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"Expected x to be a 1D vector, got shape {x.shape}.")

        cov = np.asarray(args[0])
        if cov.shape != (self.dimX, self.dimX):
            raise ValueError(
                f"Covariance shape mismatch: expected {(self.dimX, self.dimX)}, got {cov.shape}."
            )

        nr = self._aux_input()

        # Update internal state and covariance used by downstream methods.
        self.stateX = x
        self.covX = cov
        self.ne = self.num_samples

        # Draw perturbations and recenter ensemble around current state.
        enX = self.rng.multivariate_normal(self.stateX, self.covX, self.ne).T
        enX = enX - enX.mean(axis=1, keepdims=True) + self.stateX[:, None]
        enX = np.clip(enX, self.lb[:, None], self.ub[:, None])

        enF = self.function(enX, **kwargs)
        self.enX = enX
        self.enF = enF if isinstance(enF, list) else [enF]

        start_idx = 0
        nlevels = len(self.enF)
        grad_ml = np.zeros((nlevels, self.dimX))

        # Loop over levels (single level when not multilevel).
        for levelID in range(nlevels):
            dF = self.enF[levelID] - np.repeat(self.stateF, nr)
            ne = dF.shape[0]

            dx = self.enX[:, start_idx:start_idx + ne] - self.stateX[:, None]
            grad_ml[levelID] = np.squeeze(dx @ dF) / ne
            start_idx += ne

        if 'multilevel' in self.keys_en:
            weight = np.asarray(self.multilevel['ml_weights'], dtype=float)
            if weight.size > 1:
                weight_sum = np.sum(weight)
                if weight_sum != 1.0:
                    weight = weight / weight_sum
                grad = np.dot(weight, grad_ml)
            else:
                grad = grad_ml[0]
        else:
            grad = grad_ml[0]

        # Check if natural or averaged gradient (default is natural).
        if not self.keys_en.get('natural_gradient', True):
            grad = np.linalg.solve(self.covX, grad)

        return grad

    def hessian(self, x=None, *args, **kwargs):
        '''
        Ensemble-based Hessian.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, ). If None, use the last x used in gradient.
            If x is not None and it does not match the last x used in gradient, recompute the gradient first.

        args : tuple
            Additional arguments passed to function

        Returns
        -------
        hessian : ndarray
            Ensemble hessian, shape (number of controls, number of controls)

        References
        ----------
        Zhang, Y., Stordal, A.S. & Lorentzen, R.J. A natural Hessian approximation for ensemble based optimization.
        Comput Geosci 27, 355–364 (2023). https://doi.org/10.1007/s10596-022-10185-z
        '''
        # Check if self.gradient has been called with this x
        if (not np.array_equal(x, self.stateX)) and (x is not None):
            self.gradient(x, *args, **kwargs)

        nr = self._aux_input()

        # Make function ensemble to a list (for Multilevel)
        if not isinstance(self.enF, list):
            self.enF = [self.enF]

        # Define some variables for Hessian calculation
        index = 0
        nlevels = len(self.enF)
        hess_ml = np.zeros((nlevels, self.dimX, self.dimX))

        # Loop over levels (only one level if not multilevel)
        for levelID in range(nlevels):
            dF = self.enF[levelID] - np.repeat(self.stateF, nr)
            ne = self.enF[levelID].shape[0]

            # Vectorized Hessian estimate for this level.
            dx = self.enX[:, index:index + ne] - self.stateX[:, None]
            hess_ml[levelID] = (dx * dF) @ dx.T / ne - self.covX * np.mean(dF)
            index += ne

        if 'multilevel' in self.keys_en:
            weight = ot.get_list_element(self.keys_en['multilevel'], 'cov_wgt')
            weight = np.array(weight)
            if not np.sum(weight) == 1.0:
                weight = weight / np.sum(weight)
            hessian = np.sum([h*w for h, w in zip(hess_ml, weight)], axis=0)
        else:
            hessian = hess_ml[0]

        # Check if natural or averaged Hessian (default is natural)
        if not self.keys_en.get('natural_gradient', True):
            hessian = np.linalg.solve(
                self.covX,
                np.linalg.solve(self.covX, hessian).T
            ).T

        return hessian

    def calc_ensemble_weights(self, x, *args, **kwargs):
        r"""
        Calculate weights used in sequential monte carlo optimization.
        Updated version that accommodates new base class changes.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, )

        args : tuple
            Inflation factor, covariance ($C_x$, shape (number of controls, number of controls)) and survival factor

        Returns
        -------
        sens_matrix, best_ens, best_func : tuple
                The weighted ensemble, the best ensemble member, and the best objective function value
        """
        # Update state vector using new base class method
        self.stateX = x

        # Set the inflation factor, covariance and survival factor equal to the input
        self.inflation_factor = args[0]
        self.covX = args[1]
        self.survival_factor = args[2]

        # Generate ensemble of states
        if self.resample_index is None:
            self.ne = self.num_samples
        else:
            self.ne = int(np.round(self.num_samples*self.survival_factor))

        self._aux_input()

        # Generate state ensemble
        self.enX = self.rng.multivariate_normal(self.stateX, self.covX, self.ne).T

        # Truncate to bounds
        if (self.lb is not None) and (self.ub is not None):
            self.enX = np.clip(self.enX, self.lb[:, None], self.ub[:, None])

        # Evaluate objective function for ensemble
        self.enF = self.function(self.enX, **kwargs)

        if not isinstance(self.enF, list):
            self.enF = [self.enF]

        L = len(self.enF)
        if self.resample_index is None:
            self.resample_index = [None]*L

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress warnings from the weights' exponentials
            start_index = 0
            level_sens = []
            sens_matrix = np.zeros(self.enX.shape[0])
            best_ens = 0
            best_func = 0
            ml_ne_new_total = 0

            if 'multilevel' in self.keys_en.keys():
                en_size = ot.get_list_element(self.keys_en['multilevel'], 'en_size')
            else:
                en_size = [self.num_samples]

            for l in range(L):
                ml_ne = en_size[l]
                if L > 1 and l == L-1:
                    ml_ne_new = int(np.round(self.num_samples*self.survival_factor)) - ml_ne_new_total
                else:
                    ml_ne_new = int(np.round(ml_ne*self.survival_factor))  # new samples
                    ml_ne_new_total += ml_ne_new
                ml_ne_surv = ml_ne - ml_ne_new  # surviving samples

                if self.resample_index[l] is None:
                    self.particles.append(deepcopy(self.enX[:, start_index:start_index + ml_ne]))
                    self.particle_values.append(deepcopy(self.enF[l]))
                else:
                    self.particles[l][:, :ml_ne_surv] = self.particles[l][:, self.resample_index[l]]
                    self.particles[l][:, ml_ne_surv:] = deepcopy(self.enX[:, start_index:start_index + ml_ne_new])
                    self.particle_values[l][:ml_ne_surv] = self.particle_values[l][self.resample_index[l]]
                    self.particle_values[l][ml_ne_surv:] = deepcopy(self.enF[l])

                # Calculate the weights and ensemble sensitivity matrix
                weights = np.zeros(ml_ne)
                for i in range(ml_ne):
                    weights[i] = np.exp(np.clip(-(self.particle_values[l][i] - np.min(
                        self.particle_values[l])) * self.inflation_factor, None, 10))

                weights = weights + 1e-6  # Add small regularization
                weights = weights/np.sum(weights)

                level_sens.append(self.particles[l] @ weights)
                if l == L-1:  # keep the best from the finest level
                    index = np.argmin(self.particle_values[l])
                    best_ens = self.particles[l][:, index]
                    best_func = self.particle_values[l][index]
                self.resample_index[l] = self.rng.choice(ml_ne, ml_ne_surv, replace=True, p=weights)

                start_index += ml_ne_new

            if 'multilevel' in self.keys_en.keys():
                cov_wgt = ot.get_list_element(self.keys_en['multilevel'], 'cov_wgt')
                for l in range(L):
                    sens_matrix += level_sens[l]*cov_wgt[l]
                sens_matrix /= self.num_samples
            else:
                sens_matrix = level_sens[0]

            return sens_matrix, best_ens, best_func





