"""
ES-MDA type schemes
"""

# External imports
from copy import deepcopy
import numpy as np
from misc.sampling import gen_real

# Internal imports
from pipt.update_schemes.core import AssimilationScheme, StepReport, restart_options
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.full import full_update
from pipt.update_schemes.analysis.subspace import subspace_update
from pipt.update_schemes.analysis.subspace2 import subspace2_update
import pipt.misc_tools.analysis_tools as at

__all__ = ['ESMDA']

class ESMDA(AssimilationScheme):
    """Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    An iterative ensemble smoother that assimilates all data repeatedly over a
    fixed number of steps, inflating the data-error covariance at each one so
    that the repeated conditioning does not over-fit. With inflation factors
    :math:`\\alpha_i` satisfying :math:`\\sum_i 1/\\alpha_i = 1`, each step applies

    .. math::

        m \\leftarrow m + C_{md} (C_{dd} + \\alpha_i C_d)^{-1} (d_{obs} - g(m))

    with the observations re-perturbed as
    :math:`d_{obs} = d_{true} + \\sqrt{\\alpha_i} C_d^{1/2} Z`.

    The schedule is fixed rather than convergence-driven, so a run normally
    ends by exhausting its steps and reports ``success=False``. That is the
    expected outcome, not a failure.

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
    Configured through the ``mda`` block of ``keys_da``:

    ``tot_assim_steps``
        Number of assimilation steps, e.g. ``3``.
    ``inflation_param``
        Inflation factors, one per step, e.g. ``[3, 3, 3]``. Their reciprocals
        must sum to 1, which is asserted at construction. Defaults to
        ``tot_assim_steps`` repeated, which satisfies the constraint.

    Examples
    --------
    >>> result = ESMDA.assimilate(keys_da, keys_en, flow(keys_sim))
    >>> result.nit
    3

    References
    ----------
    Emerick and Reynolds, *Ensemble smoother with multiple data assimilation*
    [`emerick2013a`][].

    See Also
    --------
    ES : Single-step smoother; ES-MDA with one assimilation step.
    LMEnRML : Iterates to convergence instead of on a fixed schedule.
    """

    #: Ensemble class this scheme composes. Subclasses needing a specialised
    #: collaborator -- the multilevel variant, for instance -- override it
    #: rather than duplicating the constructor.

    COMPATIBLE_ANALYSES = {
        "approx": approx_update,
        "full": full_update,
        "subspace": subspace_update,
        "subspace2": subspace2_update,
    }

    # The perturbed observations are redrawn every step (from the ensemble's
    # stream, whose state travels with the ensemble); the misfit is scored
    # against the un-inflated draw taken at construction (`enObs_conv`).
    RESTART_ATTRIBUTES = ("enObs", "enObs_conv", "scale_data")

    def __init__(self, keys_da, keys_en, sim, analysis=None, ensemble=None):
        """Build the ensemble from the config (or take the one given) and bind the analysis.

        See the class docstring for the parameters; ``ensemble`` is a
        ready-made collaborator to run on instead of building one.
        """
        # The collaborator is handed to the scheme base, which adopts the
        # ensemble's own logger, so the log output is unchanged.
        ensemble = self.build_ensemble(keys_da, keys_en, sim, ensemble)
        # Zero tolerances switch off the base class's generic convergence
        # criteria; this scheme decides in check_convergence(). See
        # AssimilationScheme's `misfit_tol`/`step_tol` docs for why.
        super().__init__(ensemble, misfit_tol=0.0, step_tol=0.0, **restart_options(ensemble.keys_da))

        # The analysis flavour is a parameter of the algorithm, not a different
        # algorithm, so it selects an analysis object rather than a class.
        self.bind_analysis(self.resolve_analysis(analysis, ensemble.keys_da))

        self.prev_data_misfit_mean = None

        # A specialised ensemble may already have established these -- the
        # multilevel one partitions enX into per-level blocks and sets both
        # itself. Only fill
        # them in when the collaborator has not.
        if getattr(self.ensemble, 'prior_enX', None) is None:
            self.ensemble.prior_enX = deepcopy(self.enX)
        if getattr(self.ensemble, 'list_states', None) is None:
            self.ensemble.list_states = list(self.idX)
        self.ensemble.list_datatypes = self.keys_da['datatype']

        # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
        # are given as in the Simultaneous loop.
        #self.check_assimindex_simultaneous()
        #self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
        #self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(self.obs_data, self.assim_index)

        # One update per assimilation step of the MDA schedule.
        self.maxiter = len(self._ext_assim_steps())
        self.iteration = 0
        # Mirrored so ensemble-side helpers that consult the iteration
        # counter (e.g. data screening in perturb_observations) agree with
        # the scheme's, which is the one the loop advances.
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
        self.enObs_conv = deepcopy(self.enObs)

        # Get state scaling and svd of scaled prior
        self.ensemble._ext_scaling()

        # Extract the inflation parameter from MDA keyword
        self.alpha = self._ext_inflation_param()

        self.prev_data_misfit_mean = None

    # ------------------------------------------------------------------
    # AssimilationScheme contract
    # ------------------------------------------------------------------
    def update_step(self) -> StepReport:
        """Run one ES-MDA assimilation step.

        Computes the inflated analysis, forecasts the trial state, then scores
        the resulting misfit and promotes the state. Scoring after the forecast
        is what lets outlier replacement, which runs in between, feed into the
        number the scheme sees.

        Returns
        -------
        bool
            Always ``True``. ES-MDA takes a fixed number of inflated steps and
            never rejects one. The ``success`` flag it logs compares the misfit
            against the previous iteration and is a *reporting* signal only --
            returning it here would make the base class discard accepted steps.
        """
        self.calc_analysis()
        self.after_analysis()
        state = self.run_forecast(self.enX_proposal)
        self.score_and_commit()
        return StepReport(accepted=True, misfit=self.ensemble_misfit,
                          state=state)

    def check_convergence(self) -> bool:
        """ES-MDA runs its full schedule of inflated steps; nothing stops early."""
        return False

    def score(self, pred_data=None):
        """Data misfit against the *un-inflated* perturbed observations.

        ``enObs`` is redrawn each step with the covariance inflated by
        ``alpha[iteration]``, so scoring against it would compare every
        iteration to a different yardstick. ``enObs_conv`` is the copy taken
        before any inflation, which is what makes the misfit trajectory
        comparable across the schedule.
        """
        pred = self.pred_data if pred_data is None else pred_data
        return at.calc_objectivefun(
            self.enObs_conv, self._as_matrix(pred), self.cov_data
        )

    def calc_analysis(self):
        r"""
        Analysis step of ES-MDA. The analysis algorithm is similar to EnKF analysis, only difference is that the data
        covariance matrix is inflated with an inflation parameter alpha. The update is done as an iterative smoother
        where all data is assimilated at once.

        Notes
        -----
        ES-MDA is an iterative ensemble smoother with a predefined number of iterations, where the updates is done with
        the EnKF update equations but where the data covariance matrix have been inflated:

        $$ \begin{align}
        d_{obs} &= d_{true} + \sqrt{\alpha}C_d^{1/2}Z \\
        m &= m_{prior} + C_{md}(C_g + \alpha C_d)^{-1}(g(m) - d_{obs})
        \end{align} $$

        where $d_{true}$ is the true observed data, $\alpha$ is the inflation factor, $C_d$ is the data covariance
        matrix, $Z$ is a standard normal random variable, $C_{md}$ and $C_{g}$ are sample covariance matrices,
        $m$ is the model parameter, and $g(\)$ is the predicted data. Note that $\alpha$ can have a different
        value in each assimilation step and must fulfill:

        $$ \sum_{i=1}^{N_a} \frac{1}{\alpha} = 1 $$

        where $N_a$ being the total number of assimilation steps.
        """
        # Get Ensemble matrix of predicted data
        self.enPred = self.pred_data.matrix

        # The prior misfit used to be computed here, behind an `iteration == 0`
        # branch. The base scores it through `score()` before the loop now,
        # early enough for the iteration-0 artifacts to record it.
        self.data_random_state = deepcopy(np.random.get_state())
        self.enObs, self.scale_data = gen_real(
            self.vecObs,
            self.alpha[self.iteration] * self.cov_data,
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

            # Perform the update. The proposal is scheme-local, handed to
            # run_forecast and then reported back; the ensemble is only
            # written when the loop commits it.
            self.enX_proposal = self.propose_state(self.update(
                enX = self.enX,
                enY = self.enPred,
                enE = self.enObs,
                # kwargs
                prior = self.prior_enX,
                enAdj = enAdj
            ))


            # Ensure limits are respected
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX}
            self.state_layout.clip(self.enX_proposal, limits)

    def score_and_commit(self):
        """Score the forecast that followed the analysis, then commit the step.

        Was the second half of ``check_convergence``: ES-MDA never actually
        tested for convergence there, it recomputed the misfit, logged the
        iteration and promoted ``enX_temp``. Under the new contract the
        convergence question lives in :meth:`check_convergence` and this keeps
        the bookkeeping.

        Returns
        -------
        dict
            The ``why_stop`` record, also stored on ``self.why_stop``.
        """

        self.prev_data_misfit_mean = self.data_misfit_mean
        self.prev_data_misfit_std = self.data_misfit_std

        data_misfit = self.score()
        self.data_misfit_mean     = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)
        self.ensemble_misfit = data_misfit

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean),
                    'data_misfit': self.data_misfit_mean,
                    'prev_data_misfit': self.prev_data_misfit_mean}

        # Promote the trial state. Written through the ensemble so the next
        # forecast and any external reader see it.
        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        self.why_stop = why_stop
        return why_stop

    def log_columns(self, prior_run: bool = False) -> dict:
        """ES-MDA reports the inflation factor for the step just taken."""
        return {"α": "" if prior_run else self.alpha[self.iteration]}

    def _ext_inflation_param(self):
        r"""
        Extract the data covariance inflation parameter from the MDA keyword in DATAASSIM part. Also, we check that
        the criterion:

        $$ \sum_{i=1}^{N_a} \frac{1}{\alpha} = 1 $$

        is fulfilled for the inflation factor, alpha. If the keyword for inflation parameter -- INFLATION_PARAM -- is
        not provided, we set the default $\alpha_i = N_a$, where $N_a$ is the tot. no. of MDA assimilation steps (the
        criterion is fulfilled with this value).

        Returns
        -------
        alpha: list
            Data covariance inflation factor
        """
        try:
            mda_opts = dict(self.keys_da['mda'])
        except Exception:
            mda_opts = dict([self.keys_da['mda']])

        # Check if INFLATION_PARAM has been provided, and if so, extract the value(s). If not, we set alpha to the
        # default value equal to the tot. no. assim. steps
        if 'inflation_param' in mda_opts:
            alpha_tmp = mda_opts['inflation_param']
            alpha = alpha_tmp if isinstance(alpha_tmp, list) else [alpha_tmp] * len(self._ext_assim_steps())

            assert len(alpha) == len(self._ext_assim_steps()), \
            'Number of INFLATION_PARAM values does not match TOT_ASSIM_STEPS!'
        else:
            n_steps = len(self._ext_assim_steps())
            alpha = [n_steps] * n_steps

        # Check if alpha fulfills the criterion to machine precision
        assert 1 - np.finfo(float).eps <= sum(1/x for x in alpha) <= 1 + np.finfo(float).eps, \
            'Sum of inverse inflation parameters does not add up to 1!'

        return alpha

    def _ext_assim_steps(self):
        """
        Extract list of assimilation steps to perform in MDA loop from the MDA keyword (mandatory for
        MDA class) in DATAASSIM part. (This method is similar to Iterative._ext_max_iter)

        Parameters
        ----------
        keys_da : dict
            all keywords from DATAASSIM part
        mda : info
            for MDA methods

        Returns
        -------
        int
            Total number of MDA assimilation steps

        Changelog
        ---------
        - ST 7/6-16
        - ST 1/3-17: Changed to output list of assim. steps instead of just tot. assim. steps
        """
        try:
            mda_opts = dict(self.keys_da['mda'])
        except Exception:
            mda_opts = dict([self.keys_da['mda']])


        # Check if 'max_iter' has been given; if not, give error (mandatory in ITERATION)
        try:
            assim_steps = list(range(int(mda_opts['tot_assim_steps'])))
        except KeyError:
            raise AssertionError('TOT_ASSIM_STEPS has not been given in MDA!')

        # Return list assim. steps
        return assim_steps
