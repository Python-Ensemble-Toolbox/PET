"""Base ensemble for optimisation: the control vector as state, objective evaluation over the members, and multilevel bookkeeping."""
# External imports
import numpy as np
import pandas as pd


# Internal imports
from popt.misc_tools import optim_tools as ot
from ensemble import BaseEnsemble
from simulator.simple_models import noSimulation
from pipt.misc_tools.ensemble_tools import matrix_to_dict

__all__ = ['EnsembleOptimizationBase']

class EnsembleOptimizationBase(BaseEnsemble):
    '''
    Base class for the popt ensemble
    '''
    def __init__(self, options, simulator, objective):
        '''
        Parameters
        ----------
        options : dict
            Options for the ensemble class

        simulator : callable
            The forward simulator (e.g. flow). If None, no simulation is performed.

        objective : callable
            The objective function (e.g. npv)
        '''
        if simulator is None:
            sim = noSimulation({})
        else:
            sim = simulator

        # Initialize the PET Ensemble
        super().__init__(options, sim)

        # Unpack some options
        self.save_prediction = options.get('save_prediction', None)
        self.num_models      = options.get('num_models', 1) # Number of realizations for robust optimization
        self.num_samples     = self.ne # Number of perturbations for the ensemble

        # Set objective function (callable)
        if callable(objective):
            self.obj_func = objective
        else:
            raise ValueError("Objective function must be callable.")

        # Initialize state-related attributes
        self.stateX = np.array([]) # Current state vector, (nx,)
        self.stateF = None         # Function value(s) of current state
        self.bounds = []           # Bounds (untransformed) for each variable in stateX
        self.varX   = np.array([]) # Variance for state vector, (nx,)
        self.covX   = None         # Covariance matrix for state vector, (nx, nx)
        self.enX    = None         # Ensemble of state vectors ,(nx, ne)
        self.enF    = None         # Ensemble of function values, (ne,)
        self.lb     = np.array([]) # Lower bounds (transformed) for state vector, (nx,)
        self.ub     = np.array([]) # Upper bounds (transformed) for state vector, (nx,)

        # Process state information
        for name, info in self.prior_info.items():
            mean = np.asarray(info['mean'])
            var  = info['variance'] * np.ones(mean.size)
            lb, ub = info.get('limits', (-np.inf, np.inf))

            # Append to state vector and bounds
            self.stateX = np.append(self.stateX, mean)
            self.varX = np.append(self.varX, var)
            self.lb = np.append(self.lb, lb * np.ones(mean.size))
            self.ub = np.append(self.ub, ub * np.ones(mean.size))
            self.bounds += mean.size * [(lb, ub)]
            self.idX[name] = (self.stateX.size - mean.size, self.stateX.size)

        self.covX = np.diag(self.varX)  # Covariance matrix, (nx, nx)
        self.dimX = self.stateX.size    # Dimension of state vector

    def function(self, x, *args, level=None, **kwargs):
        """
        Evaluate objective values for a single state vector or an ensemble.

        Parameters
        ----------
        x : ndarray
            Control vector with shape ``(n_controls,)`` or ensemble matrix with
            shape ``(n_controls, n_perturbations)``.
        level : int, optional
            Fidelity level to simulate on, for a multilevel ensemble (``multilevel`` in
            the options). Default: a single control vector runs on the finest level and an
            ensemble is split over the levels by ``en_size``. With ``level`` given, every
            column runs on that level. Negative values count from the finest level. A value
            on a coarser level is returned but does not update ``stateF``, which stays the
            objective the optimizer works with.

        Returns
        -------
        numpy.ndarray
            Objective function values, or ``inf`` when the simulation crashed, so the
            optimizer rejects the point instead of the run ending. A crashed
            single-point evaluation leaves ``stateF`` at its last good value.

        Raises
        ------
        ValueError
            If ``x`` is not one- or two-dimensional.
        """
        self._aux_input()
        x = np.asarray(x)
        level = self._resolve_level(level) if level is not None else None

        # Check for ensemble input (nx, ne) vs. single state vector (nx,)
        ensemble_input = (x.ndim != 1)
        if ensemble_input:
            self.ne = x.shape[1]
        else:
            # A single control vector is evaluated on every geological model in
            # robust optimization: one copy per model, each paired with its own
            # model index. Without the copies the simulator got one input for
            # num_models members, and `_aux_input`'s ensemble pairing
            # ([0, 0, 1, 1, ...]) would have sent them to the wrong models.
            x = np.repeat(x[:, np.newaxis], self.num_models, axis=1)
            self.ne = self.num_models
            if self.num_models > 1:
                self.aux_input = list(range(self.num_models))

        if isinstance(self.sim, noSimulation):
            func_values = self.obj_func(x, **kwargs)
        else:
            x = self._reorganize_multilevel_ensemble(x, level=level)
            sim_success = self.calc_prediction(x, save_prediction=self.save_prediction)
            x = self._reorganize_multilevel_ensemble(x)

            if sim_success:
                func_values = self.obj_func(
                    self.sim_data,
                    input_dict=self.sim.input_dict,
                    true_order=self.sim.true_order,
                    state=matrix_to_dict(x, self.idX),
                    **kwargs
                )
            else:
                # A crashed evaluation costs the point, not the run: the optimizer
                # sees an objective it can never improve on, so backtracking rejects
                # the trial point and carries on from the last good one. Raising here
                # ended the whole optimization because one trial control vector
                # happened to be one the simulator could not run.
                self.logger.error(
                    "Simulation failed while evaluating the objective; the point is "
                    "reported as inf so the optimizer can reject it."
                )
                func_values = np.full(self.ne, np.inf)

        if ensemble_input:
            self.enF = func_values
        elif level is not None and level != self.finest_level:
            # The objective of the mean state on a coarser level belongs to the caller (the
            # gradient uses it as the reference for that level's members). `stateF` keeps the
            # finest-level value the optimizer compares trial points against.
            pass
        elif np.all(np.isfinite(func_values)):
            self.stateF = func_values
        # A crashed single-point evaluation leaves `stateF` alone. The gradient is
        # `enF - repeat(stateF, nr)`, so writing inf here would poison every later
        # gradient with inf/NaN rather than just rejecting this one point.

        return func_values


    def get_state(self):
        """
        Returns
        -------
        x : numpy.ndarray
            Control vector as ndarray, shape (number of controls, number of perturbations)
        """
        return self.stateX

    def get_cov(self):
        """
        Returns
        -------
        cov : numpy.ndarray
            Covariance matrix, shape (number of controls, number of controls)
        """
        return self.covX

    def get_bounds(self):
        """
        Returns
        -------
        bounds : list
            (min, max) pairs for each element in x. None is used to specify no bound.
        """
        return self.bounds

    def save_stateX(self, state=None, path='./', filetype='npz'):
        '''
        Save the state vector.

        Parameters
        ----------
        path : str
            Path to save the state vector. Default is current directory.

        filetype : str
            File type to save the state vector. Options are 'csv', 'npz' or 'npy'. Default is 'npz'.
        '''
        if state is None:
            stateX = self.stateX
        else:
            stateX = state

        if filetype == 'csv':
            state_dict = matrix_to_dict(stateX, self.idX)
            state_df = pd.DataFrame(data=state_dict)
            state_df.to_csv(path + 'stateX.csv', index=False)
        elif filetype == 'npz':
            state_dict = matrix_to_dict(stateX, self.idX)
            np.savez_compressed(path + 'stateX.npz', **state_dict)
        elif filetype == 'npy':
            np.save(path + 'stateX.npy', stateX)

    @property
    def num_levels(self):
        """Number of fidelity levels: the length of ``en_size``, or 1 without ``multilevel``."""
        if 'multilevel' in self.keys_en:
            return len(self.multilevel['ml_ne'])
        return 1

    @property
    def finest_level(self):
        """Index of the finest (highest-fidelity) level, the last one in ``en_size``."""
        return self.num_levels - 1

    def _resolve_level(self, level):
        """``level`` as a non-negative index, or None. Negative values count from the finest level."""
        if level is None:
            return None
        if 'multilevel' not in self.keys_en:
            raise ValueError('level is only meaningful for a multilevel ensemble (multilevel in the options)')
        level = int(level)
        n = self.num_levels
        if not -n <= level < n:
            raise ValueError(f'level {level} is out of range for {n} levels')
        return level % n

    def levels_with_members(self, x):
        """Indices of the levels that ``x`` (a matrix, or a per-level list) puts members on."""
        blocks = x if isinstance(x, list) else self._reorganize_multilevel_ensemble(x)
        if not isinstance(blocks, list):
            return [0]
        return [l for l, block in enumerate(blocks) if np.asarray(block).shape[1] > 0]

    def _reorganize_multilevel_ensemble(self, x, level=None):
        """Toggle between a state matrix and the per-level list that the multilevel forecast takes.

        A matrix becomes one block per level: an ensemble is split by ``en_size``, a single
        state (1-D or one column) goes whole to ``level`` (the finest level when None), and
        ``level`` puts all columns of an ensemble on that level. A list is joined back into a
        matrix. Without ``multilevel`` in the options ``x`` is returned unchanged.

        A single state used to be left as a matrix, which `calc_prediction` rejects for a
        multilevel ensemble; on which level a trial point ran was then up to the simulator's
        default level.
        """
        if 'multilevel' not in self.keys_en:
            return x
        ml_ne = self.multilevel['ml_ne']
        if isinstance(x, list):
            return ot.toggle_ml_state(x, ml_ne)
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if level is None and x.shape[1] > 1:
            return ot.toggle_ml_state(x, ml_ne)
        target = self.finest_level if level is None else level
        return [x if l == target else np.empty((x.shape[0], 0)) for l in range(self.num_levels)]

    def _aux_input(self):
        """
        Set the auxiliary input used for multiple geological realizations
        """

        nr = 1  # nr is the ratio of samples over models
        if self.num_models > 1:
            if np.remainder(self.num_samples, self.num_models) == 0:
                nr = int(self.num_samples / self.num_models)
                self.aux_input = list(np.repeat(np.arange(self.num_models), nr))
            else:
                raise ValueError('num_samples must be a multiple of num_models')
        return nr

