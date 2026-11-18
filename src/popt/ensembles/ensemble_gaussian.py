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

    constraint_jacobian(x, *args, **kwargs)
        Ensemble Jacobian of the constraints the objective returns

    lagrangian_hessian(x, *args, multipliers=None, **kwargs)
        Ensemble hessian of the Lagrangian

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

        # The values the members are compared with, when the caller keeps them (see `_set_reference`).
        reference = kwargs.pop('reference', None)

        self._aux_input()

        # Update internal state and covariance used by downstream methods.
        self.stateX = x
        self.covX = cov
        self.ne = self.num_samples
        if reference is not None:
            self._set_reference(reference)

        # Draw perturbations and recenter ensemble around current state.
        enX = self.rng.multivariate_normal(self.stateX, self.covX, self.ne).T
        enX = enX - enX.mean(axis=1, keepdims=True) + self.stateX[:, None]
        enX = np.clip(enX, self.lb[:, None], self.ub[:, None])

        # Reference value per level, evaluated before the ensemble so the ensemble's forecast
        # is the one left in `sim_data`.
        self.run_levels = self.levels_with_members(enX)
        self.levelF = self._level_reference_values(x, self.run_levels, **kwargs)

        enF = self.function(enX, **kwargs)
        self.enX = enX
        self.enF = enF if isinstance(enF, list) else [enF]
        if len(self.enF) != len(self.levelF):
            raise RuntimeError(f'{len(self.enF)} objective arrays returned for {len(self.levelF)} levels with members')

        # Cross-covariance of the controls and the objective, per level (single level when not multilevel).
        levels = self._level_differences()
        grad_ml = np.zeros((len(levels), self.dimX))
        for levelID, level in enumerate(levels):
            grad_ml[levelID] = np.squeeze(level['dx'] @ level['dF']) / level['n']

        grad = self._level_weights(len(levels)) @ grad_ml

        # Check if natural or averaged gradient (default is natural).
        if not self.keys_en.get('natural_gradient', True):
            grad = np.linalg.solve(self.covX, grad)

        return grad

    def constraint_jacobian(self, x=None, *args, **kwargs):
        """
        Ensemble estimate of the constraint Jacobian, from the ensemble of the last gradient.

        The objective must return its constraints with the objective (see
        ``EnsembleOptimizationBase.function``). Each row is the ensemble gradient of one
        constraint, estimated exactly as :meth:`gradient` estimates the objective's and from
        the same members, so it costs no simulations.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, ). If None, use the last x used in gradient.
            If x is not None and it does not match the last x used in gradient, recompute the gradient first.

        args : tuple
            Additional arguments passed to gradient (the covariance).

        Returns
        -------
        dict
            ``'c_eq'`` and/or ``'c_iq'``, each of shape (number of constraints, number of controls).
            Natural (covariance times Jacobian) unless ``natural_gradient`` is off.
        """
        if (x is not None) and (not np.array_equal(x, self.stateX)):
            self.gradient(x, *args, **kwargs)

        levels = self._level_differences()
        weights = self._level_weights(len(levels))

        jacobian = {}
        for key in levels[0]['dC']:
            jac_ml = np.array([level['dC'][key] @ level['dx'].T / level['n'] for level in levels])
            jac = np.tensordot(weights, jac_ml, axes=1)
            if not self.keys_en.get('natural_gradient', True):
                jac = np.linalg.solve(self.covX, jac.T).T
            jacobian[key] = jac
        return jacobian

    def sample_covariance(self):
        """
        Covariance of the perturbations of the last gradient, as sampled.

        The ensemble gradient is this matrix times the gradient, exactly for a linear
        function, so dividing by it rather than by the covariance the perturbations were
        drawn from removes the sampling error of the perturbations themselves (a
        linear-regression gradient). It is singular with fewer members than controls.

        Returns
        -------
        ndarray
            Shape (number of controls, number of controls); with levels, weighted as the gradient is.
        """
        levels = self._level_differences()
        cov_ml = np.array([level['dx'] @ level['dx'].T / level['n'] for level in levels])
        return np.tensordot(self._level_weights(len(levels)), cov_ml, axes=1)

    def _set_reference(self, reference):
        """Use ``reference`` (``'f'``, and ``'c_eq'``/``'c_iq'``) as the values at the mean state.

        ``stateF`` is the value of the latest single-state evaluation, which after a failed
        line search is a rejected trial point and not the state the ensemble is drawn around.
        An optimizer that keeps the values of its current iterate passes them here.
        """
        f = np.atleast_1d(np.asarray(reference['f'], dtype=float))
        if f.size == 1 and self.num_models > 1:
            f = np.repeat(f, self.num_models)
        self.stateF = f

        constraints = {}
        for key in ('c_eq', 'c_iq'):
            if reference.get(key) is None:
                continue
            c = np.asarray(reference[key], dtype=float)
            if c.ndim == 1:
                c = np.repeat(c[:, None], self.num_models, axis=1)
            constraints[key] = c
        self.stateC = constraints or None

    def _level_differences(self):
        """Per level with members: the perturbations and what they did to the objective and the constraints.

        Returns a list with, for each level, ``dx`` (number of controls, n), ``dF`` (n,), ``dC``
        (a dict of (number of constraints, n) arrays, empty without constraints) and ``n``, the
        number of members used. Members with a non-finite objective or constraint (a failed
        simulation) are left out of all of them.
        """
        nr = self._aux_input()

        # Make function ensemble to a list (for Multilevel)
        if not isinstance(self.enF, list):
            self.enF = [self.enF]
        enC = getattr(self, 'enC', None)
        enC = enC if isinstance(enC, list) else [enC]

        levels = []
        start_idx = 0
        for levelID, enF in enumerate(self.enF):
            dF = enF - np.repeat(self._reference(levelID), nr)
            ne = dF.shape[0]
            dC = self._constraint_differences(levelID, enC[levelID] if levelID < len(enC) else None, nr)

            # Members with non-finite objective or constraints (failed simulation/eCalc) are skipped
            finite = dF
            for c in dC.values():
                finite = np.where(np.isfinite(c).all(axis=0), finite, np.nan)
            ok = self._finite_members(finite)

            dx = self.enX[:, start_idx:start_idx + ne] - self.stateX[:, None]
            levels.append({
                'dx': dx[:, ok],
                'dF': dF[ok],
                'dC': {key: c[:, ok] for key, c in dC.items()},
                'n': ok.sum(),
            })
            start_idx += ne
        return levels

    def _constraint_differences(self, levelID, constraints, nr):
        """Constraint values of a level's members minus their values at the mean state."""
        if constraints is None:
            return {}
        refs = getattr(self, 'levelC', None)
        reference = refs[levelID] if (refs is not None and levelID < len(refs)) else getattr(self, 'stateC', None)

        differences = {}
        for key, c in constraints.items():
            if reference is not None and key in reference:
                ref = np.repeat(reference[key], nr, axis=1)
            else:
                # No value at the mean state: compare with the mean of the finite members
                ref = np.nanmean(np.where(np.isfinite(c), c, np.nan), axis=1, keepdims=True)
            differences[key] = c - ref
        return differences

    def _level_reference_values(self, x, levels, **kwargs):
        """Objective at the mean state ``x`` on each of ``levels``, the reference the members of that level are compared against.

        The finest level uses ``stateF``, the optimizer's own value at ``x``. A coarser level
        gets one extra simulation of ``x`` on that level. Subtracting the finest-level value
        from coarse members put the coarse-fine offset into every member's dF; the perturbations
        are recentred but then clipped at the bounds, so for a control at a bound the clipped
        perturbations do not sum to zero and the offset (times the mean clipped perturbation)
        appeared as a gradient component. With many controls at a bound and the gradient
        normalised by its largest element, that component set the step size and the real
        components got a fraction of it. Comparing each level with its own value at the mean
        removes the offset exactly. If the reference evaluation fails, the mean of the level's
        finite members is used instead and a warning is logged.
        """
        refs = []
        self.levelC = []  # constraint values at the mean state, level by level
        for level in levels:
            if 'multilevel' not in self.keys_en or level == self.finest_level:
                refs.append(np.asarray(self.stateF, dtype=float))
                self.levelC.append(getattr(self, 'stateC', None))
                continue
            value = np.asarray(self.function(x, level=level, **kwargs), dtype=float)
            self.levelC.append(getattr(self, 'lastC', None) if np.all(np.isfinite(value)) else None)
            if not np.all(np.isfinite(value)):
                msg = (f'Objective of the mean state on level {level} is not finite; '
                       f'the mean of the finite members on that level is used as reference instead')
                logger = getattr(self, 'logger', None)
                if logger is not None:
                    logger.warning(msg)
                else:
                    warnings.warn(msg)
                value = None
            refs.append(value)
        return refs

    def _reference(self, levelID):
        """Reference objective for the ``levelID``-th level with members: its value at the mean state from the last
        gradient, the mean of its finite members when that evaluation failed, and ``stateF`` when no gradient set it."""
        refs = getattr(self, 'levelF', None)
        if refs is None or levelID >= len(refs):
            return np.asarray(self.stateF, dtype=float)
        if refs[levelID] is not None:
            return refs[levelID]
        values = np.asarray(self.enF[levelID], dtype=float)
        return np.atleast_1d(values[np.isfinite(values)].mean())

    def _level_weights(self, nlevels):
        """Normalised weights for the ``nlevels`` levels that have members, from ``ml_weights``/``cov_wgt``."""
        if 'multilevel' not in self.keys_en:
            return np.ones(nlevels) / nlevels
        weight = np.asarray(self.multilevel['ml_weights'], dtype=float).ravel()
        if weight.size == nlevels:
            pass
        elif weight.size == self.num_levels:
            weight = weight[self.run_levels]
        else:
            raise ValueError(f'{weight.size} multilevel weights given for {nlevels} levels with members')
        return weight / np.sum(weight)

    @staticmethod
    def _finite_members(dF):
        '''Mask of ensemble members with a finite objective difference. Raises if none are finite.'''
        ok = np.isfinite(dF)
        if not ok.any():
            raise RuntimeError('Objective function is non-finite for all ensemble members')
        if not ok.all():
            print(f'Warning: skipping {np.count_nonzero(~ok)} of {ok.size} ensemble members with non-finite objective')
        return ok

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

        levels = self._level_differences()
        return self._ensemble_hessian(levels, [level['dF'] for level in levels])

    def lagrangian_hessian(self, x=None, *args, multipliers=None, detrend=True, **kwargs):
        '''
        Ensemble-based Hessian of the Lagrangian ``f - lam_eq @ c_eq - lam_iq @ c_iq``.

        The ensemble Hessian is linear in the function it is taken of, so this is
        :meth:`hessian` with the Lagrangian's change over the members in place of the
        objective's, on the ensemble of the last gradient.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, ). If None, use the last x used in gradient.
            If x is not None and it does not match the last x used in gradient, recompute the gradient first.

        args : tuple
            Additional arguments passed to gradient (the covariance).

        multipliers : dict, optional
            Lagrange multipliers ``'c_eq'`` and/or ``'c_iq'``, one per constraint. A kind left
            out counts as zero; with none this is the Hessian of the objective.

        detrend : bool, optional
            Subtract the linear trend (a least-squares fit, or the ensemble gradient times the
            perturbation with no more members than controls) from the Lagrangian's change
            over the members before forming the Hessian (default True).
            The trend contributes nothing to the Hessian on average, but its sampling error
            is larger than the curvature term by a factor of about
            ``|gradient| / (perturbation size * |Hessian|)``, which with small perturbations
            leaves nothing but noise. With False this is :meth:`hessian` applied to the Lagrangian.

        Returns
        -------
        hessian : ndarray
            Ensemble hessian, shape (number of controls, number of controls)
        '''
        if (x is not None) and (not np.array_equal(x, self.stateX)):
            self.gradient(x, *args, **kwargs)

        levels = self._level_differences()
        differences = []
        for level in levels:
            dL = level['dF']
            for key, lam in (multipliers or {}).items():
                if key in level['dC'] and lam is not None:
                    dL = dL - np.asarray(lam, dtype=float) @ level['dC'][key]
            if detrend:
                if level['n'] > self.dimX:
                    # Least-squares fit: removes a linear function exactly
                    slope = np.linalg.lstsq(level['dx'].T, dL, rcond=None)[0]
                else:
                    # Too few members to fit; with them the fit would leave no residual at all
                    slope = np.linalg.solve(self.covX, level['dx'] @ dL / level['n'])
                dL = dL - slope @ level['dx']
            differences.append(dL)
        return self._ensemble_hessian(levels, differences)

    def _ensemble_hessian(self, levels, differences):
        """The ensemble Hessian of the function whose change over the members of each level is ``differences``."""
        hess_ml = np.zeros((len(levels), self.dimX, self.dimX))

        # Loop over levels (only one level if not multilevel)
        for levelID, (level, d) in enumerate(zip(levels, differences)):
            dx = level['dx']
            # Vectorized Hessian estimate for this level.
            hess_ml[levelID] = (dx * d) @ dx.T / level['n'] - self.covX * np.mean(d)

        hessian = np.tensordot(self._level_weights(len(levels)), hess_ml, axes=1)

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





