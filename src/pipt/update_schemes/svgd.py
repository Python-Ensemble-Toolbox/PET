"""
Stein variational gradient descent (SVGD)
"""

# External imports
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial.distance import cdist

# Internal imports
from ensemble.logger import block, format_duration, heading, items
from misc.structures import StateLayout
from pipt.ensembles.adjoint_weights import DataMisfitWeights
from pipt.update_schemes.core import AssimilationScheme, StepReport, restart_options
import pipt.misc_tools.analysis_tools as at

__all__ = ['SVGD', 'DataMisfitWeights', 'EnsemblePrior', 'GraphPrior', 'fit_kernel', 'stein_direction', 'stein_terms',
           'AdaGrad', 'Adam', 'GradientDescent', 'LBFGS', 'make_optimizer']


class SVGD(AssimilationScheme):
    """Stein variational gradient descent with adjoint gradients.

    SVGD moves the members of an ensemble until they sample the posterior.
    Every member :math:`x_i` follows the direction that decreases the
    Kullback-Leibler divergence from the ensemble to the posterior the fastest
    (Stordal et al., Eq. 7):

    .. math::

        \\phi(x_i) = \\frac{1}{N} \\sum_{j=1}^N \\left[ K(x_j, x_i)\\,
            \\nabla_{x_j} \\log p(x_j \\mid y) + \\nabla_{x_j} K(x_j, x_i) \\right].

    The first term is a kernel-weighted average of the members' gradients and
    pulls the members towards high posterior density. The second term is a
    repulsion that keeps them apart, so that they spread over the posterior
    instead of collapsing onto its maximum.

    One iteration (:meth:`update_step`):

    1. **Scale.** Express the members in prior standard deviations,
       :math:`z = x / s`. Distances, repulsion and step are all taken in
       :math:`z`, so a joint state of, say, log-permeability and porosity is
       not measured in whichever has the larger numbers, and ``step_size``
       means the same for every variable.
    2. **Gradient.** Add the two terms of
       :math:`\\nabla \\log p(x_j \\mid y)` (Stordal et al., Eq. 10). The data
       term is the simulator's adjoint :math:`H^\\mathsf{T} w`, with
       :math:`w = C_d^{-1} (d_{obs} - g(x_j))`. The scheme hands the simulator
       these weights through its ``adjoint_weights``
       (:class:`DataMisfitWeights`), and the gradient comes back with the
       forecast: one forward and one adjoint run per member per iteration.
       The prior term is :math:`C_x^{-1}(\\mu - x_j)`, from a large sample of
       the prior (:class:`EnsemblePrior`).
    3. **Kernel.** Fit :math:`K(x, x') = \\exp(-(d(x, x')/\\sigma)^p)` to the
       members' current distances :math:`d` (:func:`fit_kernel`); with
       ``kernel = 'imq'`` the inverse multiquadric
       :math:`K = (1 + d^2/\\sigma^2)^{-1/2}` instead, and with
       ``kernel = 'p-imq'`` its p-variant :math:`K = (1 + (d/\\sigma)^p)^{-1/2}`.
    4. **Direction.** Compute :math:`\\phi` for every member
       (:func:`stein_direction`).
    5. **Step.** Move the members along :math:`\\phi` with the chosen
       optimizer, clip them to the prior ``limits``, and forecast.

    Every iteration is written to the log in detail: the misfits and the
    spread of the members, the two terms of the gradient, the kernel that was
    fitted, the attraction and the repulsion of the Stein direction, and the
    step. The log starts with the problem and the options, and ends with a
    summary of the run.

    The prior mean :math:`\\mu` and covariance :math:`C_x` are those of a
    large sample of the prior, ``prior_cov_samples`` realisations that are
    never simulated. :math:`C_x` is not formed: its inverse is applied through
    the singular value decomposition of the sample's anomalies. The sample
    describes at most ``prior_cov_samples - 1`` directions of the state, and
    the smallest of them may be truncated (``prior_cov_energy``). In the
    remaining directions the prior term is zero: nothing pulls the members
    back there apart from the repulsion and the prior ``limits``. A sample
    larger than the state leaves no such direction.

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
        Forward simulator returning a weighted adjoint: one gradient per
        parameter of the weighted sum of its outputs, for the weights in its
        ``adjoint_weights`` (e.g. ``subsurface.multphaseflow.jutul_darcy`` with
        ``weighted_adjoint`` set).
    ensemble : pipt.ensembles.AssimilationEnsemble, optional
        A ready-made collaborator to run on instead of building one.

    Attributes
    ----------
    kernel : str
        ``'p-kernel'``, ``'gaussian'``, ``'imq'`` or ``'p-imq'``.
    alpha : float
        Percentile level the p-kernel and the p-variant of the inverse
        multiquadric are fitted to.
    imq_level : float
        Value of the inverse multiquadric kernel at the median distance.
    optimizer : AdaGrad, Adam, GradientDescent or LBFGS
        Turns the direction into a step, and carries what it remembers of the
        earlier directions from one iteration to the next.
    prior : EnsemblePrior or None
        The prior mean and inverse covariance, in prior standard deviations;
        ``None`` when the prior term is switched off.
    kernel_p, kernel_sigma : float or None
        The kernel of the step just taken; ``None`` before the first. The
        inverse multiquadric ``'imq'`` has no :math:`p`, which stays ``None``.
    ensemble_forecasts : int
        Forecasts of the ensemble so far, the one of the prior included.
    clipped : int
        Values of the step just taken that were moved back to the prior
        ``limits``.

    Notes
    -----
    Configured through the ``svgd`` block of ``keys_da``:

    ``maxiter``
        Largest number of iterations, default ``50``. SVGD takes them all
        unless ``stein_tol`` stops it first.
    ``kernel``
        ``'p-kernel'`` (default): :math:`p` and :math:`\\sigma` are refitted every
        iteration so that the kernel is :math:`1 - \\alpha` at the
        :math:`\\alpha`-percentile of the members' distances and
        :math:`\\alpha` at the :math:`(1 - \\alpha)`-percentile (Stordal et
        al., Eqs. 24-25). ``'gaussian'``: :math:`p = 2` and :math:`\\sigma`
        follows the median rule of Liu and Wang,
        :math:`\\sigma^2 = \\mathrm{med}(d)^2 / \\log N`. ``'imq'``: the
        inverse multiquadric :math:`K = (1 + d^2/\\sigma^2)^{-1/2}` (Gorham
        and Mackey), which falls off slowly with distance and is positive
        definite in any dimension; :math:`\\sigma` is refitted every iteration
        so that the kernel is ``imq_level`` at the median distance, by default
        :math:`1/\\sqrt{N}`, which is :math:`\\sigma^2 = \\mathrm{med}(d)^2 / (N - 1)`.
        ``'p-imq'``: the p-variant of the inverse multiquadric,
        :math:`K = (1 + (d/\\sigma)^p)^{-1/2}`, with :math:`p` and
        :math:`\\sigma` refitted every iteration to the same two percentiles
        as the p-kernel. It has the p-kernel's values at those two distances
        and the slow fall-off of the inverse multiquadric beyond them; like
        the p-kernel it is positive definite only for :math:`p \\le 2`.
    ``alpha``
        Percentile level the p-kernel and ``'p-imq'`` are fitted to, default ``0.05``, the value
        of the paper's illustration of the kernel (its Fig. 1); the history
        matching example does not state one.
    ``imq_level``
        ``'imq'`` only: the value of the kernel for two members at the median
        distance from each other, between 0 and 1. It is the weight such a
        member's gradient gets in the other's direction. A high value makes
        the members share their gradients and keeps them spread; a low value
        lets every member follow its own gradient, and the ensemble narrows.
        The default is :math:`1/\\sqrt{N}` for :math:`N` members: the gradients
        of the other members then weigh as much as a member's own when they
        point in unrelated directions, since :math:`N` such vectors add up to
        :math:`\\sqrt{N}` times one of them. (The median rule of the Gaussian
        kernel gives :math:`1/N`, the balance for gradients that all point the
        same way.) Where the members are about equally far apart, as in many
        dimensions, the repulsion is then close to that of the Gaussian kernel
        for some 20 to 200 members.
    ``optimizer``
        How the direction becomes a step: ``'adagrad'`` (default,
        :class:`AdaGrad`), ``'adam'`` (:class:`Adam`), ``'gd'``
        (:class:`GradientDescent`) or ``'lbfgs'`` (:class:`LBFGS`).
    ``step_size``
        Step size of the optimizer, default ``0.1``. With ``'adagrad'`` and
        ``'adam'`` a step is about ``step_size`` prior standard deviations in
        every variable. With ``'gd'`` it multiplies the direction itself, so
        the right value depends on the size of the gradients. With ``'lbfgs'``
        it is the largest root-mean-square step, in prior standard deviations.
    ``autocorr``
        ``'adagrad'`` only: memory of the squared direction, default ``0.9``.
    ``beta1``, ``beta2``
        ``'adam'`` only: memory of the direction and of its square, defaults
        ``0.9`` and ``0.999``.
    ``memory``
        ``'lbfgs'`` only: number of earlier steps the curvature is estimated
        from, default ``10``.
    ``stein_tol``
        Stop when the root-mean-square size of the Stein direction is below
        ``stein_tol`` times its size in the first iteration, default
        ``0.001``. The direction is zero where SVGD has converged: it is what
        moves the members. On a linear-Gaussian problem with a known posterior
        the spread of the members is within 1% of the posterior's at
        ``0.001`` and within 3 to 15% at ``0.01``. An optimizer with a fixed
        step, such as ADAGRAD, may never get that low; the run then takes its
        ``maxiter`` iterations. ``0`` switches the criterion off.
    ``stein_change``
        A second condition for that stop, not used by default. When given,
        the run stops only when the size of the Stein direction is below
        ``stein_tol`` of its first size and has also changed by less than
        ``stein_change`` of itself since the iteration before: the direction
        is small and has stopped changing. A run whose direction still falls
        by more than that in every iteration goes on, also below ``stein_tol``.
    ``prior_cov_samples``
        Number of prior realisations the prior mean and covariance are
        estimated from. They are drawn from the ``prior_<name>`` blocks,
        default ``1000``; with ``importstate`` they are the first
        ``prior_cov_samples`` realisations of that file, default all of them
        (the members are its first ``ne``). ``0`` switches the prior term off.
    ``prior_cov_energy``
        Truncation of the sample's singular values, as ``energy`` elsewhere: a
        fraction, or a percentage when greater than 1. Default ``1``: keep
        every direction the sample describes.
    ``prior_cov_estimate``
        ``'sample'`` (default): the covariance of the sample
        (:class:`EnsemblePrior`). ``'graph'``: a sparse factor of the inverse
        covariance fitted to the sample on the graphs of the state
        (:class:`GraphPrior`), which has every variable that varies in the
        sample. The graphs and their reach are those of EnIF,
        ``parameter_graphs`` and ``neighbourhood_expansion`` of the ``enif``
        options, or nearest neighbours from the grid of a ``prior_<name>``
        block. ``prior_cov_energy`` is not used with it.

    The defaults -- ADAGRAD with step size 0.1 and memory 0.9, 50 iterations,
    a prior of 1000 realisations -- are the settings of the paper's
    history-matching example.

    Examples
    --------
    >>> result = SVGD.assimilate(keys_da, keys_en, jutul_darcy(keys_sim))
    >>> result.nit
    50

    References
    ----------
    Liu and Wang, *Stein variational gradient descent* [`liu2016`][];
    Stordal et al., *p-kernel Stein variational gradient descent for data
    assimilation and history matching* [`stordal2021`][];
    Gorham and Mackey, *Measuring sample quality with kernels*
    [`gorham2017`][].
    """

    KERNELS = ('p-kernel', 'gaussian', 'imq', 'p-imq')

    #: Name and title of the scheme in the log.
    NAME = "SVGD"
    TITLE = "SVGD: Stein variational gradient descent"

    #: The optimizer remembers earlier directions; a resumed run needs it back.
    #: It also needs the prior sample of the run it resumes. That sample is too
    #: large to checkpoint, so the stream state it was drawn from is kept and
    #: the sample is drawn again (see :meth:`_set_restart_state`).
    #: The size of the first Stein direction is what ``stein_tol`` compares with.
    RESTART_ATTRIBUTES = ("optimizer", "prior_stream_state", "stein_first", "stein_size")

    def __init__(self, keys_da, keys_en, sim, ensemble=None):
        """Build the ensemble from the config (or take the one given) and hand the simulator its weights.

        See the class docstring for the parameters.
        """
        ensemble = self.build_ensemble(keys_da, keys_en, sim, ensemble)
        options = dict(ensemble.keys_da.get('svgd', {}))

        # Zero tolerances switch off the base class's generic convergence
        # criteria: SVGD stops on the size of its own direction (``stein_tol``).
        super().__init__(
            ensemble,
            maxiter=int(options.get('maxiter', 50)),
            misfit_tol=0.0,
            step_tol=0.0,
            **restart_options(ensemble.keys_da)
        )

        # Store kernel and optimizer settings for the SVGD scheme.
        self.kernel = str(options.get('kernel', 'p-kernel')).lower()
        self.alpha = float(options.get('alpha', 0.05))
        level = options.get('imq_level', None)
        self.imq_level = 1 / np.sqrt(self.ne) if level is None else float(level)
        self.optimizer = make_optimizer(options)
        self.ensemble_forecasts = 0
        self.clipped = 0
        self._started = time.time()
        self.prior_cov_samples = options.get('prior_cov_samples')
        self.prior_cov_energy = float(options.get('prior_cov_energy', 1.0))
        self.prior_cov_estimate = str(options.get('prior_cov_estimate', 'sample')).lower()
        self.kernel_p = None
        self.kernel_sigma = None
        self.stein_tol = float(options.get('stein_tol', 0.001))
        self.stein_change = options.get('stein_change')
        if self.stein_change is not None:
            self.stein_change = float(self.stein_change)
        self.stein_first = None        # size of the Stein direction in the first iteration
        self.stein_size = None         # its size in the iteration last started
        self.stein_ratio = None        # its size now, relative to that
        self.stein_converged = False
        self._check_requirements()

        # Set ensemble-related attributes
        self.ensemble.prior_enX = self.enX.copy()
        self.ensemble.list_states = list(self.idX)
        self.ensemble.list_datatypes = self.keys_da['datatype']
        self.ensemble._ext_scaling()   # sets state_scaling, the prior standard deviations

        # The prior mean and inverse covariance, from a large sample of the prior.
        self.prior_stream_state = self.ensemble.rng.get_state()
        self.prior = self._build_prior()

        # Set up the data covariance and the simulator's adjoint weights.
        self.cov_data = self._data_covariance()
        self.sim.adjoint_weights = DataMisfitWeights(
            self.data_layout, self.obs_vector, self.cov_data,
            position=self.ensemble._record_positions(),
            scale=(self.data_df.scale_min, self.data_df.scale_max) if self.data_df.is_scaled else None,
        )

    def _check_requirements(self):
        """Refuse options and setups the scheme cannot run on."""
        if self.kernel not in self.KERNELS:
            raise ValueError(f"svgd kernel must be 'p-kernel', 'gaussian', 'imq' or 'p-imq', not {self.kernel!r}")
        if not 0 < self.alpha < 0.5:
            raise ValueError(f"svgd alpha must lie in (0, 0.5), not {self.alpha}")
        if not 0 <= self.stein_tol < 1:
            raise ValueError(f"svgd stein_tol must lie in [0, 1), not {self.stein_tol}")
        if self.stein_change is not None and not self.stein_change > 0:
            raise ValueError(f"svgd stein_change must be positive, not {self.stein_change}")
        if not 0 < self.imq_level < 1:
            raise ValueError(f"svgd imq_level must lie in (0, 1), not {self.imq_level}")
        if self.ne < 3:
            raise ValueError(f"SVGD needs at least 3 members to fit its kernel, not {self.ne}")
        if getattr(self.sim, 'weighted_parameters', None) is None:
            raise ValueError(
                "SVGD needs the simulator's weighted adjoint: set 'weighted_adjoint' "
                "(the parameters to differentiate) in the simulator section"
            )
        if getattr(self.ensemble, 'sparse_info', None):
            raise ValueError("SVGD does not support compressed data ('compress') yet")
        if self.prior_cov_samples is not None and int(self.prior_cov_samples) in (1, 2):
            raise ValueError(
                f"svgd prior_cov_samples must be 0 (no prior term) or at least 3, not {self.prior_cov_samples}"
            )
        if not 0 < self.prior_cov_energy <= 100:
            raise ValueError(f"svgd prior_cov_energy must lie in (0, 100], not {self.prior_cov_energy}")
        if self.prior_cov_estimate not in ('sample', 'graph'):
            raise ValueError(f"svgd prior_cov_estimate must be 'sample' or 'graph', not {self.prior_cov_estimate!r}")

    def _data_covariance(self):
        """The data covariance ``(nd, nd)``, or the variance ``(nd,)`` when the errors are uncorrelated."""
        cov_data = getattr(self.ensemble, 'cov_data', None)
        if cov_data is not None:
            return cov_data
        if np.ndim(self.obs_variance) != 1:
            raise ValueError("SVGD needs the data variance itself, not an error ensemble ('emp_cov')")
        return self.obs_variance

    # ------------------------------------------------------------------
    # The prior
    # ------------------------------------------------------------------
    def _build_prior(self):
        """The prior mean and inverse covariance in prior standard deviations; ``None`` when switched off."""
        if self.prior_cov_samples is not None and int(self.prior_cov_samples) == 0:
            return None
        samples = self._prior_samples() / self.state_scaling[:, None]
        nx, n = samples.shape
        if self.prior_cov_estimate == 'graph':
            # The graphs and the number of hops are EnIF's, from the same place in the configuration.
            from pipt.update_schemes.analysis.enif import parameter_graph
            groups = sorted(self.idX.items(), key=lambda item: item[1][0])
            options = self.keys_da.get('enif', {})
            graphs = [((start, stop), parameter_graph(name, stop - start, options, self.prior_info))
                      for name, (start, stop) in groups]
            hops = int(options.get('neighbourhood_expansion', 2))
            prior = GraphPrior(samples, graphs, hops=hops)
            message = (f"SVGD prior: a sparse factor of the inverse covariance fitted to {n} realisations on the "
                       f"graphs of the state ({hops} hops), for {prior.rank} of {nx} variables.")
            if prior.rank < nx:
                message += " The others do not vary in the realisations and have no prior term."
            self.logger(message)
            return prior
        prior = EnsemblePrior(samples, energy=self.prior_cov_energy)

        message = f"SVGD prior: {n} realisations, {prior.rank} of {nx} directions of the state resolved."
        if prior.rank < nx:
            message += " The prior term is zero in the others."
        self.logger(message)
        return prior

    def _prior_samples(self) -> np.ndarray:
        """The large sample of the prior, ``(nx, prior_cov_samples)``, in the state's own units.

        With ``importstate`` the realisations of that file, whose first ``ne``
        are the members. Otherwise drawn from the ``prior_<name>`` blocks, on
        a stream of its own that starts where the ensemble's stream stood, so
        drawing leaves the ensemble's stream untouched and can be repeated.
        """
        n = None if self.prior_cov_samples is None else int(self.prior_cov_samples)
        keys_en = self.ensemble.keys_en

        if 'importstate' in keys_en:
            file = np.load(keys_en['importstate'], allow_pickle=True)
            samples, layout = StateLayout.from_dict({key: file[key] for key in file.files}, ne=n)
            if n is not None and samples.shape[1] < n:
                raise ValueError(
                    f"svgd prior_cov_samples is {n}, but {keys_en['importstate']!r} "
                    f"holds {samples.shape[1]} realisations"
                )
        else:
            stream = np.random.RandomState()
            stream.set_state(self.prior_stream_state)
            samples, layout = StateLayout.from_prior_info(self.prior_info, 1000 if n is None else n,
                                                          rng=stream, save=False)

        if samples is None or dict(layout.indices) != dict(self.idX):
            raise ValueError(
                "SVGD could not draw its prior sample from the 'prior_<name>' blocks in the layout of the "
                "state; describe the prior there, or set 'prior_cov_samples = 0' to run without the prior term"
            )
        return samples

    def _set_restart_state(self, state: dict) -> None:
        """Restore the optimizer, and draw the prior sample of the interrupted run again."""
        super()._set_restart_state(state)
        self.prior = self._build_prior()

    # ------------------------------------------------------------------
    # AssimilationScheme contract
    # ------------------------------------------------------------------
    def run_assimilation(self):
        """Write the problem and the options to the log, then run (:meth:`AssimilationScheme.run_assimilation`)."""
        self._started = time.time()
        options = list(self.ensemble.keys_da.get('svgd', {}).items())
        self._log(block(self.TITLE, [], heavy=True)[:2])
        self._log(block("Problem", self._problem_rows()) + [""])
        self._log(block("User-specified options", options or [("(none, all defaults)", "")]) + [""])
        return super().run_assimilation()

    def update_step(self) -> StepReport:
        """Move every member one step along the Stein direction, then forecast.

        The direction uses the gradients of the forecast the members already
        have, so one forward and one adjoint run per member per iteration.
        """
        # 1-2. Members and their gradients in prior standard deviations, z = x / s.
        scale = self.state_scaling[:, None]
        Z = self.enX / scale
        gradient = self.log_posterior_gradient(Z)

        # 3-4. The kernel for the members' current spread, and the direction it gives.
        distances = cdist(Z.T, Z.T)
        self.kernel_p, self.kernel_sigma = fit_kernel(distances, self.kernel, self.alpha, self.imq_level)
        attraction, repulsion, K = stein_terms(Z, gradient, distances, self.kernel_p, self.kernel_sigma, self.kernel)
        direction = (attraction + repulsion) / Z.shape[1]

        stop = self.stein_stop(_rms(direction))

        self._log([heading(self._iteration_title())])
        self._log(items("Members", self._member_rows(Z)) + [""])
        self._log(items("Gradient of the log-posterior", self._gradient_rows(gradient)) + [""])
        self._log(items("Kernel", self._kernel_rows(distances, K)) + [""])
        self._log(items("Stein direction", [
            ("Attraction, rms", f"{_rms(attraction) / Z.shape[1]:.3e}"),
            ("Repulsion, rms", f"{_rms(repulsion) / Z.shape[1]:.3e}"),
            ("Direction, rms", f"{_rms(direction):.3e}"),
            ("Direction, rms relative to the first iteration", f"{self.stein_ratio:.3e}"),
        ]) + [""])
        if stop is not None:
            self._log([f"Stopped: {self.conv_msg}"])
            return stop

        # 5. Step, back in the state's own units and within the prior limits.
        step = self.optimizer.step(direction, Z)
        before = [
            ("Optimizer", type(self.optimizer).__name__),
            ("Step size", f"{self.optimizer.step_size:g}"),
            ("Step, rms (prior sd)", f"{_rms(step):.3e}"),
            ("Step, largest (prior sd)", f"{np.abs(step).max():.3e}"),
        ]
        width = len("Data misfit of the moved members, mean")
        self._log(items("Step", before, width))

        self.after_analysis()
        state, misfit = self._forecast_step(Z, step)
        self._log(items(None, [
            ("Values moved back to the limits", self.clipped),
            ("Data misfit of the moved members, mean", f"{np.mean(misfit):.3e}"),
        ], width))

        self.prev_data_misfit_mean = self.data_misfit_mean
        return StepReport(accepted=True, misfit=misfit, state=state)

    def stein_stop(self, size: float) -> StepReport | None:
        """The report that ends the run when the Stein direction has become small, else ``None``.

        ``size`` is the size of the direction of the members as they are now.
        It is compared with the size in the first iteration; the ratio is kept
        in ``stein_ratio``. With ``stein_change`` the size must also have
        changed by less than that part of itself since the iteration before.
        Nothing has been simulated when this is called, so the members and
        their forecast stay as they are.
        """
        previous, self.stein_size = self.stein_size, float(size)
        if self.stein_first is None:
            self.stein_first = float(size)
        self.stein_ratio = float(size) / self.stein_first if self.stein_first > 0 else 0.0
        if self.stein_ratio >= self.stein_tol:
            return None
        reason = (f"The Stein direction is {self.stein_ratio:.2e} of its size in the first iteration, "
                  f"below stein_tol = {self.stein_tol:g}")
        if self.stein_change is not None:
            moved = abs(size - previous) / previous if previous else np.inf
            if not moved < self.stein_change:      # small, but still changing: the run goes on
                return None
            reason += (f", and it changed by {moved:.2g} of its size since the iteration before, "
                       f"below stein_change = {self.stein_change:g}")
        self.stein_converged = True
        self.conv_msg = reason
        return StepReport(accepted=False, misfit=self.ensemble_misfit, state=None, why_stop={"stein_tol": True})

    def check_convergence(self) -> bool:
        """``True`` once :meth:`stein_stop` has ended the run."""
        return self.stein_converged

    def run_forecast(self, state):
        """Forecast ``state`` and count the forecast (:meth:`AssimilationScheme.run_forecast`)."""
        self.ensemble_forecasts += 1
        return super().run_forecast(state)

    def after_loop(self, converged: bool) -> None:
        """Save what the base class saves, then write the summary of the run to the log."""
        super().after_loop(converged)
        reason = self.conv_msg or "no stopping reason recorded"
        self._log([""] + block(f"{self.NAME} finished: {reason}", self._summary_rows(), heavy=True))

    def _forecast_step(self, Z: np.ndarray, step: np.ndarray):
        """Forecast the members ``Z`` moved by ``step``, clipped to the prior limits; the state and its misfits."""
        proposal = (Z + step) * self.state_scaling[:, None]
        unclipped = proposal.copy()
        self.state_layout.clip(proposal, self._limits())
        self.clipped = int(np.count_nonzero(proposal != unclipped))
        state = self.run_forecast(proposal)
        return state, self.score()

    def score(self, pred_data=None):
        """Data misfit :math:`(d_{obs} - g(x))^\\mathsf{T} C_d^{-1} (d_{obs} - g(x))` against the unperturbed observations."""
        pred = self._as_matrix(self.pred_data if pred_data is None else pred_data)
        obs = np.repeat(self.obs_vector[:, None], pred.shape[1], axis=1)
        return at.calc_objectivefun(obs, pred, self.cov_data)

    def log_columns(self, prior_run: bool = False) -> dict:
        """SVGD reports the Stein direction the step just taken started from, and its kernel.

        ``Φ (%)`` is the size of that direction in percent of its size in the
        first iteration, the number ``stein_tol`` is compared with.
        """
        if prior_run:
            return {"Φ (%)": "", "p": "", "σ": ""}
        p = "" if self.kernel_p is None else f"{self.kernel_p:.3g}"     # the inverse multiquadric has none
        return {"Φ (%)": self.stein_percent(), "p": p, "σ": f"{self.kernel_sigma:.3g}"}

    def stein_percent(self) -> str:
        """The size of the Stein direction in percent of its size in the first iteration, as the run table shows it."""
        return f"{100 * self.stein_ratio:.3g}"

    # ------------------------------------------------------------------
    # The log
    # ------------------------------------------------------------------
    # The rows are ``(label, value)`` pairs for ``ensemble.logger.items``, or
    # ``(label, symbol, value)`` triples for ``ensemble.logger.block``. A
    # scheme built on this one reuses them and adds its own.
    def _log(self, lines) -> None:
        """Write ``lines`` to the log, one message each."""
        if self.logger is None:
            return
        for line in lines:
            self.logger(line)

    def _iteration_title(self) -> str:
        """The heading of the iteration about to be taken, with the wall time so far."""
        return f"Iteration {self.iteration + 1}  ({format_duration(time.time() - self._started)})"

    def _problem_rows(self) -> list:
        """The problem and the settings of the scheme, for the block at the start of the log."""
        nx = self.enX.shape[0]
        rows = [
            ("State", "", ", ".join(str(name) for name in self.idX)),
            ("State variables", "nx", nx),
            ("Members", "ne", self.ne),
            ("Data", "nd", self.obs_vector.size),
            ("Prior term", "", "off" if self.prior is None else f"{self.prior.rank} of {nx} directions resolved"),
            ("Kernel", "", self.kernel),
        ]
        if self.kernel in ('p-kernel', 'p-imq'):
            rows.append(("Percentile level of the kernel", "α", f"{self.alpha:g}"))
        if self.kernel == 'imq':
            rows.append(("Kernel at the median distance", "", f"{self.imq_level:g}"))
        return rows + self._step_rows()

    def _change_rows(self) -> list:
        """The second condition of the stop on the Stein direction, when there is one."""
        if self.stein_change is None:
            return []
        return [("and has changed by less than this part of its size", "", f"{self.stein_change:g}")]

    def _step_rows(self) -> list:
        """How a direction becomes a step, for the block at the start of the log."""
        return [
            ("Optimizer", "", type(self.optimizer).__name__),
            ("Step size", "ε", f"{self.optimizer.step_size:g}"),
            ("Iterations, at most", "", self.maxiter),
            ("Stop when the Stein direction is below this part of its first size", "", f"{self.stein_tol:g}"),
            *self._change_rows(),
            ("Forecasts of the ensemble per iteration", "", 1),
        ]

    def _member_rows(self, Z: np.ndarray) -> list:
        """Misfits and spread of the members ``Z``."""
        misfit = np.asarray(self.ensemble_misfit, dtype=float)
        rows = [
            ("Data misfit, mean", f"{misfit.mean():.3e}"),
            ("Data misfit, smallest .. largest", f"{misfit.min():.3e} .. {misfit.max():.3e}"),
        ]
        if self.prior is not None:
            rows.append(("Prior misfit, mean", f"{self.prior.misfit(Z).mean():.3e}"))
        rows.append(("Spread, rms standard deviation (prior sd)", f"{_rms(Z.std(axis=1, ddof=1)):.3g}"))
        return rows

    def _gradient_rows(self, gradient: np.ndarray) -> list:
        """Size of the two terms of ``gradient``, the gradient of the log-posterior at the members."""
        data_term = self.adjoints * self.state_scaling[:, None]
        rows = [("Data term, rms", f"{_rms(data_term):.3e}")]
        if self.prior is not None:
            rows.append(("Prior term, rms", f"{_rms(gradient - data_term):.3e}"))
        return rows

    def _kernel_rows(self, distances: np.ndarray, K: np.ndarray) -> list:
        """The kernel fitted to ``distances``, and the weight ``K`` gives every member."""
        n = K.shape[0]
        apart = distances[np.triu_indices(n, 1)]
        weight = K.sum(axis=0) / n
        rows = [("Distances, smallest .. median .. largest",
                 f"{apart.min():.3g} .. {np.median(apart):.3g} .. {apart.max():.3g}")]
        if self.kernel_p is not None:           # the inverse multiquadric has none
            rows.append(("Exponent p", f"{self.kernel_p:.3g}"))
        rows += [
            ("Bandwidth σ", f"{self.kernel_sigma:.3g}"),
            ("Kernel weight of a member, smallest .. largest", f"{weight.min():.3g} .. {weight.max():.3g}"),
            ("Kernel weight, mean", f"{weight.mean():.3g}"),
        ]
        return rows

    def _summary_rows(self) -> list:
        """The run in numbers, for the block at the end of the log."""
        rows = []
        if self.prior_data_misfit_mean is not None and self.data_misfit_mean is not None:
            rows += [
                ("Data misfit of the prior, mean", "", f"{self.prior_data_misfit_mean:.6e}"),
                ("Data misfit at the end, mean", "", f"{self.data_misfit_mean:.6e}"),
                ("Relative change", "", f"{100 * (self.data_misfit_mean / self.prior_data_misfit_mean - 1):.2f}%"),
            ]
        return rows + [
            ("Iterations", "", self.iteration),
            ("Forecasts of the ensemble", "", self.ensemble_forecasts),
            ("Wall time", "", format_duration(time.time() - self._started)),
        ]

    # ------------------------------------------------------------------
    # The gradient
    # ------------------------------------------------------------------
    def log_posterior_gradient(self, Z: np.ndarray) -> np.ndarray:
        """:math:`\\nabla_z \\log p(z \\mid y)` at the members ``Z``, ``(nx, ne)`` (Stordal et al., Eq. 10).

        Everything is in prior standard deviations, :math:`z = x / s`. The
        data term :math:`H^\\mathsf{T} C_d^{-1} (d_{obs} - g(x))` is the
        weighted adjoint that came back with the members' forecast, which is a
        gradient in :math:`x`; the chain rule :math:`\\nabla_z = s\\,\\nabla_x`
        carries it to :math:`z`.
        """
        data_gradient = self.adjoints * self.state_scaling[:, None]
        return data_gradient + self.prior_gradient(Z)

    def prior_gradient(self, Z: np.ndarray):
        """The prior term :math:`C_z^{-1}(\\mu_z - z)` at the members ``Z``; zero when the prior is switched off."""
        if self.prior is None:
            return 0.0
        return self.prior.gradient(Z)

    def _limits(self) -> dict:
        """The ``(lower, upper)`` prior limits of each state variable; ``None`` where there is none."""
        return {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX}


class EnsemblePrior:
    """A Gaussian prior known through a sample of it: applies :math:`C^{-1}` without forming :math:`C`.

    The sample ``(nx, n)`` gives the mean :math:`\\mu` and the anomalies
    :math:`A = (X - \\mu 1^\\mathsf{T}) / \\sqrt{n - 1}`, for which
    :math:`C \\approx A A^\\mathsf{T}`. With the singular value decomposition
    :math:`A = U \\Sigma V^\\mathsf{T}`,

    .. math::

        \\nabla \\log p(x) = C^{-1} (\\mu - x)
            \\approx U \\Sigma^{-2} U^\\mathsf{T} (\\mu - x).

    Only :math:`U` ``(nx, rank)`` and :math:`\\Sigma` are kept. This is the
    inverse within the directions :math:`U` the sample describes, at most
    ``n - 1`` of them; in every other direction the gradient is zero. A
    sample larger than the state describes them all, and the gradient is then
    that of the sample covariance.

    Parameters
    ----------
    samples : np.ndarray
        Realisations of the prior, ``(nx, n)``.
    energy : float
        Truncation of the singular values, as in
        :func:`pipt.misc_tools.analysis_tools.truncSVD`; ``1`` keeps them all.
        Singular values at rounding level are always dropped: they are
        directions without variance, which cannot be inverted.

    Attributes
    ----------
    mean : np.ndarray
        :math:`\\mu`, ``(nx, 1)``.
    U : np.ndarray
        :math:`U`, ``(nx, rank)``.
    S : np.ndarray
        The singular values, the diagonal of :math:`\\Sigma`, ``(rank,)``; their
        squares are the variances along the columns of ``U``.
    """

    def __init__(self, samples: np.ndarray, energy: float = 1.0):
        n = samples.shape[1]
        if n < 2:
            raise ValueError(f"a prior covariance needs at least 2 realisations, not {n}")
        self.mean = samples.mean(axis=1, keepdims=True)
        anomalies = (samples - self.mean) / np.sqrt(n - 1)
        U, S, _ = at.truncSVD(anomalies, energy=energy)

        rounding = S[0] * max(anomalies.shape) * np.finfo(float).eps
        resolved = S > rounding
        self.U = U[:, resolved]
        self.S = S[resolved]

    @property
    def rank(self) -> int:
        """Number of directions the prior term acts in."""
        return self.S.size

    @property
    def L(self) -> np.ndarray:
        """The factor :math:`L = U \\Sigma` of :math:`C = L L^\\mathsf{T}`, ``(nx, rank)``."""
        return self.U * self.S

    @property
    def L_inv(self) -> np.ndarray:
        """Its pseudo-inverse :math:`\\Sigma^{-1} U^\\mathsf{T}`, ``(rank, nx)``: whitens a deviation from the mean."""
        return (self.U / self.S).T

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """:math:`\\nabla \\log p` at every column of ``X``, ``(nx, ne)``."""
        coefficients = self.U.T @ (self.mean - X)             # (rank, ne)
        return self.U @ (coefficients / self.S[:, None] ** 2)

    def misfit(self, X: np.ndarray) -> np.ndarray:
        """:math:`(x - \\mu)^\\mathsf{T} C^{-1} (x - \\mu)` for every column of ``X``, ``(ne,)``.

        Minus twice the log-density up to a constant, so that
        :meth:`gradient` is minus half its gradient. Like the gradient it
        counts only the directions the sample describes.
        """
        coefficients = self.U.T @ (X - self.mean)             # (rank, ne)
        return np.sum((coefficients / self.S[:, None]) ** 2, axis=0)


class GraphPrior:
    """A Gaussian prior known through a sample of it and a graph of its variables: a sparse factor of :math:`C^{-1}`.

    Where :class:`EnsemblePrior` takes the covariance of the sample as it is,
    which has at most ``n - 1`` directions, this prior fits a sparse matrix
    :math:`C_w` with :math:`C^{-1} \\approx C_w^\\mathsf{T} C_w` to the sample.
    A row of :math:`C_w` has entries only for the variables within ``hops``
    steps of its own in the graph, so the fit has few unknowns per row and
    gives an inverse covariance of full rank from a sample smaller than the
    state. The fit is that of EnIF, ``optimize_sparse_affine_kr_map`` of
    ``graphite-maps``, which :func:`graphite_maps.precision_estimation.fit_precision_cholesky_approximate`
    calls before it multiplies the factor with its transpose; here the factor
    itself is kept, because it whitens: :math:`C_w (x - \\mu)` has the identity
    as covariance.

    Each group of variables is fitted on its own, so the groups are
    independent in this prior. A variable that does not vary in the sample is
    left out: it has no prior term and no row in the factor.

    Parameters
    ----------
    samples : np.ndarray
        Realisations of the prior, ``(nx, n)``.
    graphs : list
        ``((start, stop), graph)`` for every group of rows of the state, with
        ``graph`` a :class:`networkx.Graph` on the nodes ``0 .. stop - start - 1``.
    hops : int
        Reach of a row of the factor in its graph, EnIF's
        ``neighbourhood_expansion``.

    Attributes
    ----------
    mean : np.ndarray
        :math:`\\mu`, ``(nx, 1)``.
    rows : np.ndarray
        The rows of the state that vary in the sample, ``(rank,)``.
    L_inv : scipy.sparse.csr_array
        :math:`L^{-1}`, ``(rank, nx)``: :math:`C_w` on the rows that vary. It
        whitens a deviation from the mean.
    L
        :math:`L` with :math:`C = L L^\\mathsf{T}`, ``(nx, rank)``, as something
        to multiply with: ``L @ M`` and ``L.T @ M`` are solves with
        :math:`C_w`, which is factorised once.
    """

    def __init__(self, samples: np.ndarray, graphs: list, hops: int = 2):
        import networkx as nx
        from graphite_maps.precision_estimation import optimize_sparse_affine_kr_map

        if samples.shape[1] < 2:
            raise ValueError(f"a prior covariance needs at least 2 realisations, not {samples.shape[1]}")
        if hops < 1:
            raise ValueError(f"the factor of the prior must reach at least 1 hop in the graph, not {hops}")
        self.mean = samples.mean(axis=1, keepdims=True)
        varies = np.ptp(samples, axis=1) > 0
        blocks, rows = [], []
        for (start, stop), graph in graphs:
            active = np.flatnonzero(varies[start:stop])
            if active.size == 0:
                continue
            # As fit_precision_cholesky_approximate: the graph reached in ``hops`` steps, and every node itself.
            reach = nx.convert_node_labels_to_integers(graph.subgraph(active), ordering='sorted')
            reach = nx.power(reach, k=hops)
            reach.add_edges_from((node, node) for node in reach.nodes())
            blocks.append(optimize_sparse_affine_kr_map(U=samples[start:stop][active].T, G=reach, use_tqdm=False))
            rows.append(start + active)
        if not blocks:
            raise ValueError("no variable of the state varies in the prior sample")
        self.rows = np.concatenate(rows)
        factor = sparse.csc_array(sparse.block_diag(blocks, format='csc'))
        rank = self.rows.size
        select = sparse.csr_array((np.ones(rank), (np.arange(rank), self.rows)), shape=(rank, samples.shape[0]))
        self.L_inv = sparse.csr_array(factor @ select)
        self.L = _FactorSolve(factor, select)

    @property
    def rank(self) -> int:
        """Number of variables the prior term acts on."""
        return self.rows.size

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """:math:`\\nabla \\log p` at every column of ``X``, ``(nx, ne)``."""
        return self.L_inv.T @ (self.L_inv @ (self.mean - X))

    def misfit(self, X: np.ndarray) -> np.ndarray:
        """:math:`(x - \\mu)^\\mathsf{T} C^{-1} (x - \\mu)` for every column of ``X``, ``(ne,)``."""
        return np.sum((self.L_inv @ (X - self.mean)) ** 2, axis=0)


class _FactorSolve:
    """:math:`L = P^\\mathsf{T} C_w^{-1}` of :class:`GraphPrior`, or its transpose, as something to multiply with.

    :math:`P` picks the rows of the state the factor has. ``self @ M`` solves
    with :math:`C_w`, whose LU factorisation is made at the first product and
    shared with the transpose. It is not pickled.
    """

    def __init__(self, factor, select, transposed: bool = False, cache: dict = None):
        self.factor, self.select, self.transposed = factor, select, transposed
        self._cache = {} if cache is None else cache

    @property
    def shape(self) -> tuple:
        return self.select.shape if self.transposed else self.select.shape[::-1]

    @property
    def T(self) -> '_FactorSolve':
        return _FactorSolve(self.factor, self.select, not self.transposed, self._cache)

    def _lu(self):
        if 'lu' not in self._cache:
            self._cache['lu'] = splu(sparse.csc_matrix(self.factor))
        return self._cache['lu']

    def __matmul__(self, M):
        M = np.asarray(M, dtype=float)
        if self.transposed:                                   # L' M = C_w^-T P M
            return self._lu().solve(np.ascontiguousarray(self.select @ M), trans='T')
        return self.select.T @ self._lu().solve(np.ascontiguousarray(M))      # L M = P' C_w^-1 M

    def __getstate__(self):
        return {**self.__dict__, '_cache': {}}


# ---------------------------------------------------------------------- #
# The kernel and the Stein direction
# ---------------------------------------------------------------------- #
def fit_kernel(D: np.ndarray, kernel: str = 'p-kernel', alpha: float = 0.05, level: float = None) -> tuple:
    """The kernel's ``(p, sigma)`` for the members' distances ``D``.

    The p-kernel is fitted so that :math:`K = 1 - \\alpha` at the
    :math:`\\alpha`-percentile of the distances and :math:`K = \\alpha` at the
    :math:`(1 - \\alpha)`-percentile (Stordal et al., Eq. 25); the Gaussian
    kernel has :math:`p = 2` and the median rule. Only distinct pairs count:
    a member replaced by a copy of another adds a zero that says nothing about
    the spread.

    The inverse multiquadric :math:`K = (1 + d^2/\\sigma^2)^{-1/2}`
    (``kernel = 'imq'``) has no :math:`p`: ``None`` is returned in its place,
    which is how :func:`stein_direction` knows the kernel. Its :math:`\\sigma`
    makes the kernel ``level`` at the median distance :math:`m`,
    :math:`\\sigma = m\\,\\mathrm{level} / \\sqrt{1 - \\mathrm{level}^2}`. Without a
    ``level`` it is :math:`1/\\sqrt{n}` for :math:`n` members, so
    :math:`\\sigma = m / \\sqrt{n - 1}`.

    The p-variant of the inverse multiquadric,
    :math:`K = (1 + (d/\\sigma)^p)^{-1/2}` (``kernel = 'p-imq'``), is fitted
    as the p-kernel is, to :math:`K = 1 - \\alpha` at the
    :math:`\\alpha`-percentile :math:`d_{lo}` of the distances and
    :math:`K = \\alpha` at the :math:`(1 - \\alpha)`-percentile :math:`d_{hi}`.
    The two conditions are :math:`(d_{lo}/\\sigma)^p = (1 - \\alpha)^{-2} - 1`
    and :math:`(d_{hi}/\\sigma)^p = \\alpha^{-2} - 1`, so

    .. math::

        p = \\frac{\\log\\big((\\alpha^{-2} - 1) / ((1 - \\alpha)^{-2} - 1)\\big)}{\\log(d_{hi}/d_{lo})},
        \\qquad \\sigma = d_{hi}\\,(\\alpha^{-2} - 1)^{-1/p} .

    For the same distances this :math:`p` is a fixed multiple of the
    p-kernel's, 2.02 for :math:`\\alpha = 0.05`. :func:`stein_direction` has to
    be told that the pair belongs to this kernel (its ``kernel``).
    """
    d = D[np.triu_indices_from(D, k=1)]
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("all members coincide; the kernel cannot be fitted")
    if kernel == 'gaussian':
        n = D.shape[0]
        return 2.0, float(np.median(d) / np.sqrt(np.log(n)))
    if kernel == 'imq':
        if level is None:
            level = 1 / np.sqrt(D.shape[0])
        return None, float(np.median(d) * level / np.sqrt(1 - level ** 2))
    d_lo, d_hi = np.quantile(d, [alpha, 1 - alpha])
    if not d_hi > d_lo:
        raise ValueError(
            f"the members' distances have no spread between their {alpha:g} and "
            f"{1 - alpha:g} percentiles; the p-kernel cannot be fitted"
        )
    if kernel == 'p-imq':
        at_lo, at_hi = (1 - alpha) ** -2 - 1, alpha ** -2 - 1      # (d / sigma)^p at the two percentiles
        p = np.log(at_hi / at_lo) / np.log(d_hi / d_lo)
        return float(p), float(d_hi / at_hi ** (1 / p))
    p = np.log(np.log(alpha) / np.log(1 - alpha)) / np.log(d_hi / d_lo)
    sigma = d_hi / (-np.log(alpha)) ** (1 / p)
    return float(p), float(sigma)


def stein_direction(Z: np.ndarray, gradient: np.ndarray, D: np.ndarray, p: float, sigma: float,
                    kernel: str = None) -> np.ndarray:
    """The SVGD direction of every member, ``(nx, ne)`` (Stordal et al., Eq. 7).

    Parameters
    ----------
    Z : np.ndarray
        Members, ``(nx, ne)``.
    gradient : np.ndarray
        :math:`\\nabla \\log p` at each member, ``(nx, ne)``.
    D : np.ndarray
        Their distances, ``(ne, ne)``.
    p, sigma : float
        The kernel :math:`K = \\exp(-(D/\\sigma)^p)`; with ``p = None`` the
        inverse multiquadric :math:`K = (1 + D^2/\\sigma^2)^{-1/2}`
        (:func:`fit_kernel`).
    kernel : str, optional
        ``'p-imq'`` when ``p`` and ``sigma`` are those of the p-variant of
        the inverse multiquadric, :math:`K = (1 + (D/\\sigma)^p)^{-1/2}`. For
        the other kernels it need not be given: ``p`` says which one it is.
    """
    attraction, repulsion, _ = stein_terms(Z, gradient, D, p, sigma, kernel)
    return (attraction + repulsion) / Z.shape[1]


def stein_terms(Z: np.ndarray, gradient: np.ndarray, D: np.ndarray, p: float, sigma: float,
                kernel: str = None) -> tuple:
    """The two terms of the SVGD direction before they are added and divided by ``ne``, and the kernel matrix.

    The arguments are those of :func:`stein_direction`.

    Returns
    -------
    attraction : np.ndarray
        :math:`\\sum_j K(z_j, z_i)\\, \\nabla \\log p(z_j)` for every member, ``(nx, ne)``.
    repulsion : np.ndarray
        :math:`\\sum_j \\nabla_{z_j} K(z_j, z_i)` for every member, ``(nx, ne)``.
    K : np.ndarray
        The kernel matrix, ``(ne, ne)``.
    """
    # The kernel, and the c of its gradient: grad_{z_j} K(z_j, z_i) = c_ij (z_i - z_j).
    # Members that coincide do not repel each other, the diagonal included:
    # c_ij = 0 where D_ij = 0.
    if kernel == 'p-imq':
        # K = (1 + u)^(-1/2) with u = (D / sigma)^p, and c = (p / 2) K^3 u / D^2. As K^2 = 1 / (1 + u), the product
        # K^3 u is K (1 - K^2), which stays finite when p is in the hundreds and u overflows.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            K = 1 / np.sqrt(1 + (D / sigma) ** p)
            c = np.where(D > 0, 0.5 * p * K * (1 - K ** 2) / D ** 2, 0.0)
    elif p is None:
        K = 1 / np.sqrt(1 + (D / sigma) ** 2)
        c = np.where(D > 0, K ** 3 / sigma ** 2, 0.0)
    else:
        u = (D / sigma) ** p
        K = np.exp(-u)
        # c = K p D^(p-2) / sigma^p, written as p K u / D^2. In many dimensions the members are almost
        # equally far apart and p is in the hundreds, so that D^(p-2) and sigma^p overflow on their own,
        # while u is of order one. Where K has fallen to zero, so has c, whatever u is.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            c = np.where((D > 0) & (K > 0), p * K * u / D ** 2, 0.0)

    # Attraction: sum_j K_ji gradient_j.
    attraction = gradient @ K

    # Repulsion: sum_j grad_{z_j} K(z_j, z_i) = sum_j c_ij (z_i - z_j).
    repulsion = Z * c.sum(axis=0) - Z @ c

    return attraction, repulsion, K


def _rms(array: np.ndarray) -> float:
    """Root-mean-square size of the entries of ``array``."""
    return float(np.sqrt(np.mean(array ** 2)))


# ---------------------------------------------------------------------- #
# Optimizers: how the direction becomes a step
# ---------------------------------------------------------------------- #
# An optimizer has one method, ``step(direction, members)``, returning the
# increment to add to the members. The direction is the one that decreases the
# Kullback-Leibler divergence, so the step is along it, not against it. Both
# arguments and the increment are ``(nx, ne)``. ``members`` is where the
# members stand now; only L-BFGS needs it. An optimizer must be picklable,
# since it is checkpointed with the run.
class GradientDescent:
    """Plain gradient descent on the Kullback-Leibler divergence, the update of Stordal et al., Eq. 7.

    .. math::

        \\Delta_k = \\epsilon\\, \\phi_k

    The step is *along* :math:`\\phi_k`: the Stein direction already points
    the way the divergence decreases, which is up the log-posterior and so
    down the data misfit.

    Parameters
    ----------
    step_size : float
        :math:`\\epsilon`. It multiplies the direction itself, so the right
        value depends on the size of the gradients.
    """

    def __init__(self, step_size):
        self.step_size = step_size

    def step(self, direction: np.ndarray, members=None) -> np.ndarray:
        """The increment for ``direction``."""
        return self.step_size * direction


class AdaGrad:
    """ADAGRAD with a running mean of the squared direction, as in Liu and Wang's code.

    .. math::

        G_k = \\rho\\, G_{k-1} + (1 - \\rho)\\, \\phi_k^2, \\qquad
        \\Delta_k = \\epsilon\\, \\frac{\\phi_k}{\\delta + \\sqrt{G_k}},

    starting from :math:`G_1 = \\phi_1^2`. Dividing by the typical size of the
    direction makes a step about :math:`\\epsilon` in every variable, however
    large its gradient is.

    Parameters
    ----------
    step_size : float
        :math:`\\epsilon`.
    autocorr : float
        :math:`\\rho`, the memory of the squared direction.

    Attributes
    ----------
    mean_square : np.ndarray or None
        :math:`G_k`; ``None`` before the first step.
    """

    #: :math:`\\delta`, keeps the division finite where the direction is zero.
    FUDGE = 1e-6

    def __init__(self, step_size, autocorr=0.9):
        if not 0 <= autocorr < 1:
            raise ValueError(f"svgd autocorr must lie in [0, 1), not {autocorr}")
        self.step_size = step_size
        self.autocorr = autocorr
        self.mean_square = None

    def step(self, direction: np.ndarray, members=None) -> np.ndarray:
        """The increment for ``direction``; updates the running mean."""
        if self.mean_square is None:
            self.mean_square = direction ** 2
        else:
            self.mean_square = self.autocorr * self.mean_square + (1 - self.autocorr) * direction ** 2
        return self.step_size * direction / (self.FUDGE + np.sqrt(self.mean_square))


class Adam:
    """Adam (Kingma and Ba): ADAGRAD's scaling applied to a running mean of the direction.

    .. math::

        m_k = \\beta_1 m_{k-1} + (1 - \\beta_1)\\, \\phi_k, \\qquad
        v_k = \\beta_2 v_{k-1} + (1 - \\beta_2)\\, \\phi_k^2, \\qquad
        \\Delta_k = \\epsilon\\, \\frac{\\hat m_k}{\\delta + \\sqrt{\\hat v_k}},

    starting from zero, with :math:`\\hat m_k = m_k / (1 - \\beta_1^k)` and
    :math:`\\hat v_k = v_k / (1 - \\beta_2^k)` correcting for that start.

    Parameters
    ----------
    step_size : float
        :math:`\\epsilon`.
    beta1, beta2 : float
        Memory of the direction and of its square.

    Attributes
    ----------
    mean, mean_square : np.ndarray or float
        :math:`m_k` and :math:`v_k`.
    count : int
        Steps taken, :math:`k`.
    """

    #: :math:`\\delta`, keeps the division finite where the direction is zero.
    FUDGE = 1e-8

    def __init__(self, step_size, beta1=0.9, beta2=0.999):
        for name, beta in (('beta1', beta1), ('beta2', beta2)):
            if not 0 <= beta < 1:
                raise ValueError(f"svgd {name} must lie in [0, 1), not {beta}")
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.mean = 0.0
        self.mean_square = 0.0
        self.count = 0

    def step(self, direction: np.ndarray, members=None) -> np.ndarray:
        """The increment for ``direction``; updates the running means."""
        self.count += 1
        self.mean = self.beta1 * self.mean + (1 - self.beta1) * direction
        self.mean_square = self.beta2 * self.mean_square + (1 - self.beta2) * direction ** 2
        mean = self.mean / (1 - self.beta1 ** self.count)
        mean_square = self.mean_square / (1 - self.beta2 ** self.count)
        return self.step_size * mean / (self.FUDGE + np.sqrt(mean_square))


class LBFGS:
    """Limited-memory BFGS on the whole ensemble.

    The Stein direction :math:`\\phi` takes the place of minus a gradient, and
    all members are treated as one long vector :math:`z`. From the last
    ``memory`` steps the optimizer keeps the pairs

    .. math::

        s_k = z_{k+1} - z_k, \\qquad y_k = -(\\phi_{k+1} - \\phi_k),

    how far the members moved and how much the direction changed. They define
    an approximation :math:`B_k` of the inverse Hessian, applied with the
    usual two-loop recursion, and the quasi-Newton step is

    .. math::

        \\Delta_k = B_k\\, \\phi_k.

    Two things differ from textbook L-BFGS, because :math:`\\phi` is not the
    gradient of a function the optimizer can evaluate:

    - There is no line search. The step is shortened so that its
      root-mean-square size never exceeds ``step_size``, and the first step,
      taken before any pair exists, has exactly that size.
    - A pair can have :math:`s_k^\\mathsf{T} y_k \\le 0`, which no positive
      definite :math:`B_k` can reproduce. Such pairs are skipped.

    :math:`B_k` is therefore not the inverse Hessian of anything, and
    :math:`\\Delta_k` need not point the way the divergence decreases.
    Nothing checks that it does: the cap on the step is the only safeguard.

    The pairs are ``2 * memory`` arrays of the size of the ensemble, and they
    are part of the restart checkpoint.

    Parameters
    ----------
    step_size : float
        Largest root-mean-square step.
    memory : int
        Number of pairs kept.

    Attributes
    ----------
    pairs : list
        ``(s, y, 1 / (s @ y))`` of the steps remembered, oldest first.
    skipped : int
        Pairs dropped because their curvature :math:`s^\\mathsf{T} y` was not positive.
    """

    #: A pair is kept when ``s @ y`` exceeds this fraction of ``|s| |y|``.
    CURVATURE_TOL = 1e-10

    def __init__(self, step_size, memory=10):
        if memory < 1:
            raise ValueError(f"svgd memory must be at least 1, not {memory}")
        self.step_size = step_size
        self.memory = int(memory)
        self.pairs = []
        self.skipped = 0
        self._last_members = None
        self._last_direction = None

    def step(self, direction: np.ndarray, members: np.ndarray) -> np.ndarray:
        """The increment for ``direction`` at ``members``; remembers the step that led here."""
        self._remember(direction, members)
        step = self._apply_inverse_hessian(direction.ravel())

        size = np.sqrt(np.mean(step ** 2))
        if size > 0 and (size > self.step_size or not self.pairs):
            step *= self.step_size / size
        return step.reshape(direction.shape)

    def reset(self) -> None:
        """Forget the pairs. The next one is still learned from the step the members take now."""
        self.pairs = []

    def _remember(self, direction: np.ndarray, members: np.ndarray) -> None:
        """Add the pair of the step that led to ``members``, if its curvature is positive."""
        if self._last_members is not None:
            s = (members - self._last_members).ravel()
            y = (self._last_direction - direction).ravel()
            curvature = s @ y
            if curvature > self.CURVATURE_TOL * np.linalg.norm(s) * np.linalg.norm(y):
                self.pairs.append((s, y, 1.0 / curvature))
                del self.pairs[:-self.memory]
            else:
                self.skipped += 1
        self._last_members = members.copy()
        self._last_direction = direction.copy()

    def _apply_inverse_hessian(self, vector: np.ndarray) -> np.ndarray:
        """:math:`B_k` times ``vector``, by the two-loop recursion; the vector itself while no pair exists."""
        q = vector.copy()
        alphas = []
        for s, y, rho in reversed(self.pairs):
            alpha = rho * (s @ q)
            alphas.append(alpha)
            q -= alpha * y

        if self.pairs:
            s, y, _ = self.pairs[-1]
            q *= (s @ y) / (y @ y)        # the usual first guess of the inverse Hessian

        for (s, y, rho), alpha in zip(self.pairs, reversed(alphas)):
            q += (alpha - rho * (y @ q)) * s
        return q


def make_optimizer(options: dict):
    """The optimizer the ``svgd`` options ask for; ADAGRAD unless ``optimizer`` names another.

    Every optimizer reads ``step_size``; ``autocorr`` belongs to ``'adagrad'``,
    ``beta1`` and ``beta2`` to ``'adam'``, and ``memory`` to ``'lbfgs'``.
    """
    name = str(options.get('optimizer', 'adagrad')).lower()
    step_size = float(options.get('step_size', 0.1))
    if step_size <= 0:
        raise ValueError(f"svgd step_size must be positive, not {step_size}")

    if name == 'adagrad':
        return AdaGrad(step_size, autocorr=float(options.get('autocorr', 0.9)))
    if name == 'adam':
        return Adam(step_size, beta1=float(options.get('beta1', 0.9)), beta2=float(options.get('beta2', 0.999)))
    if name == 'gd':
        return GradientDescent(step_size)
    if name == 'lbfgs':
        return LBFGS(step_size, memory=int(options.get('memory', 10)))
    raise ValueError(f"svgd optimizer must be 'adagrad', 'adam', 'gd' or 'lbfgs', not {name!r}")
