"""
EnKF type schemes
"""
# External imports
import numpy as np
from copy import deepcopy
from misc.sampling import gen_real

# Internal imports
from pipt.update_schemes.core import AssimilationScheme, StepReport, restart_options
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.subspace import subspace_update
# Misc. tools used in analysis schemes
import pipt.misc_tools.extract_tools as extract



class EnKF(AssimilationScheme):
    """Ensemble Kalman Filter (EnKF).

    Assimilates data sequentially, updating the state once per group of
    observations in the order given by ``assimindex``. Each update applies the
    Kalman equations with the covariances approximated from the ensemble:

    .. math::

        m \\leftarrow m + C_{md} (C_{dd} + C_d)^{-1} (d_{obs} - g(m))

    There is no damping and no rejection: every step is accepted, and the run
    ends once the data groups are exhausted.

    Parameters
    ----------
    keys_da : dict
        Parsed ``dataassim`` configuration. Besides the keys every scheme
        reads -- ``data``, ``datavar``, ``obsname``, ``truedataindex`` -- the
        ones this scheme acts on are listed under Notes.
    keys_en : dict
        Parsed ``ensemble`` configuration: ensemble size ``ne``, the ``state``
        variable names, and the ``prior_<name>`` blocks describing each.
    sim : object
        Forward simulator instance, e.g. ``simulator.opm.flow``.
    analysis : {'approx', 'full', 'subspace'}, optional
        Analysis flavour, i.e. how the ensemble-approximated sensitivity is
        inverted. Defaults to the ``analysis`` key in ``keys_da``, falling back
        to ``'approx'``. The flavours differ in cost and in how they handle a
        rank-deficient ensemble; they solve the same update equation.

    Attributes
    ----------
    ensemble : pipt.ensembles.AssimilationEnsemble
        Collaborator holding the state realisations, observed data and
        simulator. Its state is exposed as properties on the scheme, so
        ``scheme.enX`` and ``scheme.keys_da`` read straight through.
    analysis : pipt.update_schemes.analysis.AnalysisBase
        The bound analysis object. Note the constructor takes ``analysis`` as
        a *name* and this attribute holds the resulting object, the way
        ``Model(optimizer="adam").optimizer`` is an optimizer instance.
    analysis_name : str
        The flavour name that was resolved, e.g. ``'approx'``.
    iteration : int
        Accepted iterations completed so far.
    data_misfit, prior_data_misfit : float
        Current and initial mean data misfit.

    Notes
    -----
    ``assimindex`` determines the grouping and ordering of the sequential
    updates. If all data are to be assimilated in a single step, use :class:`ES`,
    which is this scheme specialised to one group.

    ``energy`` sets the fraction of singular values retained in the truncated
    SVD (default 0.98); values above 1 are read as percentages.

    Every data group is assimilated exactly once, so the prior-increment term
    that distinguishes ``full`` from ``approx`` is never reached: ``"full"``
    is pointed at the same class as ``"approx"`` in
    :attr:`COMPATIBLE_ANALYSES`. :class:`ES` inherits this.

    Examples
    --------
    >>> result = EnKF.assimilate(keys_da, keys_en, flow(keys_sim))

    References
    ----------
    Evensen, *Data Assimilation: The Ensemble Kalman Filter* [`evensen2009a`][].

    See Also
    --------
    ES : All-data-at-once form of the same update.
    """

    # Neither this class nor ES revisit a data group, so the prior-increment
    # term "full" adds over "approx" never applies -- the two produce
    # identical output (pinned by the characterisation suite), just through
    # more expensive machinery for "full". Rather than special-case that in
    # code, "full" is simply pointed at the same class as "approx" here.
    COMPATIBLE_ANALYSES = {
        "approx": approx_update,
        "full": approx_update,
        "subspace": subspace_update,
    }

    RESTART_ATTRIBUTES = ("enObs", "enObs_conv", "scale_data")

    def __init__(self, keys_da, keys_en, sim, analysis=None, ensemble=None):
        """Build the ensemble from the config and bind the analysis.

        See the class docstring for the parameters.
        """
        # Build the collaborator, then hand it to the scheme base -- which
        # adopts the ensemble's own logger, so log output is unchanged.
        ensemble = self.build_ensemble(keys_da, keys_en, sim, ensemble)
        # Zero tolerances switch off the base class's generic convergence
        # criteria; this scheme decides in check_convergence(). See
        # AssimilationScheme's `misfit_tol`/`step_tol` docs for why.
        super().__init__(ensemble, misfit_tol=0.0, step_tol=0.0, **restart_options(ensemble.keys_da))

        # Flavour is a parameter, so it selects an analysis object not a class.
        self.bind_analysis(self.resolve_analysis(analysis, ensemble.keys_da))

        self.prev_data_misfit_mean = None

        self.ensemble.prior_enX = deepcopy(self.enX)
        self.ensemble.list_states = list(self.idX.keys())

        # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
        # are given as in the Simultaneous loop.
        self.ensemble.check_assimindex_simultaneous()

        self.ensemble.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
        self.ensemble.list_datatypes = self.keys_da['datatype']


        # One update per assimilation index.
        self.maxiter = len(self.keys_da['assimindex'])
        self.iteration = 0
        # Mirrored for ensemble-side helpers that consult it.
        self.ensemble.iteration = 0
        self.lam = 0  # set LM lamda to zero as we are doing one full update.

        if 'energy' in self.keys_da:
            # initial energy (Remember to extract this)
            self.trunc_energy = self.keys_da['energy']
            if self.trunc_energy > 1:  # ensure that it is given as percentage
                self.trunc_energy /= 100.
        else:
            self.trunc_energy = 0.98

        # Get the perturbed observations and observation scaling
        self.vecObs = self.ensemble.obs_vector
        self.enObs = self.ensemble.perturb_observations(self.vecObs)
        self.ensemble._ext_scaling()

    def calc_analysis(self):
        """
        Calculate the analysis step of the EnKF procedure. The updating is done using the Kalman filter equations, using
        svd for numerical stability. Localization is available.
        """
        # Augment observed and predicted data
        if extract.is_enabled(self.keys_da.get('emp_cov', False)):
            self.enPred = self.pred_data.matrix
        else:
            self.enPred = self.pred_data.matrix

            #self.cov_data = at.gen_covdata(
            #    self.datavar,
            #    self.assim_index,
            #    self.list_datatypes
           # )
            self.cov_data = self.ensemble.obs_variance

            self.data_random_state = deepcopy(np.random.get_state())
            self.enObs, self.scale_data = gen_real(
                self.vecObs,
                self.cov_data,
                self.ne,
                rng=self.ensemble.rng,
                return_chol=True
            )

        self.E = np.dot(self.enObs, self.proj)

        if 'localanalysis' in self.keys_da:
            self.ensemble.local_analysis_update()
            # The one path that still writes ensemble.enX_temp, which nothing
            # reads now -- so take its result explicitly.
            proposed = getattr(self.ensemble, "enX_temp", None)
            self.enX_proposal = self.enX if proposed is None else proposed
        else:
            # Check for adjoint
            if hasattr(self, 'adjoints'):
                enAdj = self.adjoints   # (nd, nx, ne), None without adjoints
            else:
                enAdj = None

            self.enX_proposal = self.propose_state(self.update(
                enX = self.enX,
                enY = self.enPred,
                enE = self.enObs,
                prior = self.prior_enX,
                enAdj = enAdj
            ))

            # Ensure limits are respected
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX.keys()}
            self.state_layout.clip(self.enX_proposal, limits)

    # ------------------------------------------------------------------
    # AssimilationScheme contract
    # ------------------------------------------------------------------
    def update_step(self) -> StepReport:
        """Run one EnKF step: analysis, forecast, then score and commit.

        Returns
        -------
        bool
            Always ``True``. The EnKF applies one update per data group and
            has no rejection path.
        """
        self.calc_analysis()
        self.after_analysis()
        state = self.run_forecast(self.enX_proposal)
        self.score_and_commit()
        return StepReport(accepted=True, misfit=self.ensemble_misfit,
                          state=state)

    def check_convergence(self) -> bool:
        """The EnKF runs its full sweep of data groups; nothing stops early."""
        return False

    def score_and_commit(self):
        """
        Calculate the "convergence" of the method. Important to
        """
        self.prev_data_misfit_mean = self.prior_data_misfit_mean

        # only calulate for the final (posterior) estimate
        if self.iteration + 1 == len(self.keys_da['assimindex']):
            data_misfit = self.score()
            self.ensemble_misfit = data_misfit
            self.data_misfit_mean = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

        else:  # sequential updates not finished. Misfit is not relevant
            self.data_misfit_mean = self.prior_data_misfit_mean

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean),
                    'data_misfit': self.data_misfit_mean,
                    'prev_data_misfit': self.prev_data_misfit_mean}

        # Update state ensemble

        if self.data_misfit_mean == self.prev_data_misfit_mean:
            self.logger.info(
                f'EnKF update {self.iteration} complete!')
        else:
            if self.data_misfit_mean < self.prior_data_misfit_mean:
                self.logger.info(
                    f'EnKF update complete! Objective function decreased from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}.')
            else:
                self.logger.info(
                    f'EnKF update complete! Objective function increased from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}.')
        self.why_stop = why_stop
        return why_stop
