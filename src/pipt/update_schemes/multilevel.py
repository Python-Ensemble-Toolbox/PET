'''
Multilevel schemes developed in the 4DSeis project.

The multilevel machinery is *ensemble* work: it reorganises the state into one
block per fidelity level and configures the simulator to run them. It therefore
lives on :class:`MultilevelEnsemble`, which the scheme composes, rather than
being inherited by the scheme itself.

That split matters. ``multilevel`` previously subclassed the ensemble and
``esmda_hybrid`` inherited from both it and the ES-MDA scheme, relying on C3
linearisation to route ``super().__init__()`` into the scheme's constructor.
Once the schemes stopped inheriting the ensemble, the ensemble intercepted that
chain and the scheme's ``__init__`` silently stopped running -- leaving
``alpha`` unset and the analysis step broken. Composition removes the ordering
dependence entirely.
'''

#──────────────────────────────────────────────────────────────────────────────────────
from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.esmda import ESMDA
from pipt.update_schemes.analysis.base import AnalysisResult
from pipt.misc_tools import analysis_tools as at
from misc.sampling import gen_real
from pipt.update_schemes.analysis.hybrid import hybrid_update

import numpy as np
from copy import deepcopy
#──────────────────────────────────────────────────────────────────────────────────────


__all__ = ['MultilevelEnsemble', 'multilevel', 'esmda_hybrid']


class MultilevelEnsemble(Ensemble):
    """Ensemble whose state is partitioned into fidelity levels.

    ``enX`` is a *list* of matrices, one per level, rather than a single
    ``(nx, ne)`` matrix, and the simulator is configured to run each level.
    Everything else is the ordinary assimilation ensemble.

    Attributes
    ----------
    enX : list of ndarray
        State ensemble per level; ``enX[l]`` has shape ``(nx, ml_ne[l])``.
    tot_level : int
        Number of fidelity levels.
    ml_ne : list of int
        Ensemble size at each level.
    """

    def __init__(self, keys_da, keys_en, sim):
        super().__init__(keys_da, keys_en, sim)

        self.list_states = list(self.idX.keys())

        # Keep the unpartitioned prior: state scaling is defined over the whole
        # state, not per level. Under the previous class layout the scheme's
        # __init__ ran before the split and so saw the matrix; holding it here
        # reproduces that without depending on constructor ordering.
        self._flat_prior_enX = deepcopy(self.enX)

        # Reorganize prior ensemble to multilevel structure if nested is true
        self.enX = self.reorganize_ml_prior(self.enX)
        self.prior_enX = deepcopy(self.enX)

        # Set ML specific options for simulator
        self._init_sim()

        self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
        self.list_datatypes = self.keys_da['datatype']

        self.cov_data = self.obs_variance
        self.vecObs = self.obs_vector

    def _ext_scaling(self):
        """Compute state scaling from the unpartitioned prior.

        Once the prior is a list of per-level blocks, the scaling is still
        defined over the whole state matrix.
        """
        self.state_scaling = at.calc_scaling(
            self._flat_prior_enX, self.idX, self.prior_info
        )
        self.Am = None

    def _init_sim(self):
        """
        Ensure that the simulator is initiallized to handle ML forward simulation.
        """
        self.sim.multilevel = [l for l in range(self.tot_level)]
        self.sim.rawmap = [None] * self.tot_level
        self.sim.ecl_coarse = [None] * self.tot_level
        self.sim.well_cells = [None] * self.tot_level

    def reorganize_ml_prior(self, enX: np.ndarray) -> list:
        '''
        Reorganize prior ensemble to multilevel structure (list of matrices).
        '''
        ml_enX = []
        start  = 0
        for l in self.multilevel['levels']:
            stop = start + self.multilevel['ml_ne'][l]
            ml_enX.append(enX[:, start:stop])
            start = stop
        return ml_enX


#: Historical name for the multilevel container, which used to be what schemes
#: inherited. It is the ensemble now, so this is an alias rather than a base.
multilevel = MultilevelEnsemble


class esmda_hybrid(ESMDA):
    '''
    A multilevel implementation of the ES-MDA algorithm with the hybrid gain.

    Composes a :class:`MultilevelEnsemble` and binds ``hybrid_update`` for the
    per-level gain, the same way :class:`~pipt.update_schemes.esmda.ESMDA`
    binds ``approx_update`` and friends. It is not just ``ESMDA`` with an
    extra flavour, though: its own ``COMPATIBLE_ANALYSES`` offers only
    ``"hybrid"``, deliberately narrower than ``ESMDA``'s -- ``approx_update``
    et al. expect a single ``enX``/``proj`` matrix, and this scheme's state is
    partitioned into one such matrix *per level*, which those analyses were
    never written to handle.

    Notes
    -----
    Requires a ``multilevel`` block in ``keys_en`` giving ``levels``,
    ``en_size`` per level and ``ml_weights``.
    '''

    ENSEMBLE_CLASS = MultilevelEnsemble
    COMPATIBLE_ANALYSES = {"hybrid": hybrid_update}

    def __init__(self, keys_da, keys_en, sim, analysis=None, ensemble=None):
        super().__init__(keys_da, keys_en, sim, analysis=analysis, ensemble=ensemble)

        self.proj = []
        for l in range(self.tot_level):
            nl = self.ml_ne[l]
            proj_l = (np.eye(nl) - np.ones((nl, nl))/nl) / np.sqrt(nl - 1)
            self.proj.append(proj_l)

    # update_step() and check_convergence() are inherited from ESMDA unchanged:
    # the multilevel variant differs in the analysis and the scoring, not in
    # the step choreography or the fixed schedule.

    def score(self, pred_data=None):
        """Data misfit over every fidelity level at once.

        ``pred_data`` is one frame per level here, so the levels are
        concatenated along the ensemble axis and scored as a single ensemble
        against the un-inflated perturbations, as
        :meth:`pipt.update_schemes.esmda.ESMDA.score` does for one level.
        """
        pred = self.pred_data if pred_data is None else pred_data
        levels = [self._as_matrix(frame) for frame in pred]
        return at.calc_objectivefun(
            self.enObs_conv, np.concatenate(levels, axis=1), self.cov_data
        )

    def calc_analysis(self):
        """The ES-MDA analysis over every fidelity level: per-level predictions, redrawn observations, the hybrid update, clipped proposals."""

        # Get ensemble predictions at all levels
        self.enPred = []
        for l in range(self.tot_level):
            enPred_level = self.pred_data[l].matrix
            self.enPred.append(enPred_level)

        # Initialize GeoStat class for generating realizations

        if self.iteration == 0:  # first iteration

            self.data_random_state = deepcopy(np.random.get_state())

            self.ml_enObs = []
            self.scale_data = []
            self.E = []
            for l in range(self.tot_level):

                # Generate real data and scale data
                enObs_level, scale_data_level = gen_real(
                    self.vecObs,
                    self.alpha[self.iteration] * self.cov_data,
                    self.ml_ne[l],
                    rng=self.ensemble.rng,
                    return_chol=True
                )
                self.ml_enObs.append(enObs_level)
                self.scale_data.append(scale_data_level)
                self.E.append(np.dot(enObs_level, self.proj[l]))

        else:
            self.data_random_state = deepcopy(np.random.get_state())

            for l in range(self.tot_level):
                self.ml_enObs[l], self.scale_data[l] = gen_real(
                    self.vecObs,
                    self.alpha[self.iteration] * self.cov_data,
                    self.ml_ne[l],
                    rng=self.ensemble.rng,
                    return_chol=True
                )
                self.E[l] = np.dot(self.ml_enObs[l], self.proj[l])

        # Calculate the update step: one state-space step per fidelity level.
        result = AnalysisResult.coerce(self.update(
            enX = self.enX,
            enY = self.enPred,
            enE = self.ml_enObs
        ))
        self.step = result.step
        limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX}
        # A scheme-local proposal, one entry per fidelity level.
        enX_proposal = []
        for l in range(self.tot_level):
            level = self.enX[l] + self.step[l]
            self.state_layout.clip(level, limits)
            enX_proposal.append(level)
        self.enX_proposal = enX_proposal

    def score_and_commit(self):
        """Score the forecast that followed the analysis, then commit the step.

        Was the second half of ``check_convergence``. ES-MDA never tested for
        convergence there; it recomputed the misfit, logged the iteration and
        promoted ``enX_temp``.

        Returns
        -------
        dict
            The ``why_stop`` record, also stored on ``self.why_stop``.
        """

        self.prev_data_misfit_mean = self.data_misfit_mean
        self.prev_data_misfit_std = self.data_misfit_std

        data_misfit = self.score()
        self.ensemble_misfit = data_misfit
        self.data_misfit_mean = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean),
                    'data_misfit': self.data_misfit_mean,
                    'prev_data_misfit': self.prev_data_misfit_mean}

        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        self.why_stop = why_stop
        return why_stop
