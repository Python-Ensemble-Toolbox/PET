"""
ES type schemes
"""
from pipt.update_schemes.enkf import EnKF

import numpy as np


class ES(EnKF):
    """Ensemble Smoother (ES).

    Assimilates all observations simultaneously in a single update, rather than
    sequentially in time as the filter does. It is :class:`EnKF` specialised to
    one data group, and shares its analysis step; only the iteration budget and
    the misfit bookkeeping differ.

    A single conditioning step is cheap but can over-correct when the model is
    strongly non-linear. :class:`ESMDA` addresses this by spreading the same
    update over several inflated steps.

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
    ``assimindex`` is flattened to a single group at construction, so the
    ordering that matters for :class:`EnKF` has no effect here.

    Because there is only one step, the ``full`` flavour coincides with
    ``approx`` -- the prior-increment term they differ over is only reached
    when iterating -- so :attr:`EnKF.COMPATIBLE_ANALYSES`, inherited
    unchanged here, points ``"full"`` at the cheaper ``approx`` analysis.

    Examples
    --------
    >>> result = ES.assimilate(keys_da, keys_en, flow(keys_sim))
    >>> result.nit
    1

    References
    ----------
    Evensen, *Data Assimilation: The Ensemble Kalman Filter* [`evensen2009a`][].

    See Also
    --------
    EnKF : Sequential form of the same update.
    ESMDA : Spreads the conditioning over several inflated steps.
    """

    def __init__(self, keys_da, keys_en, sim, analysis=None, ensemble=None):
        """Build the ensemble from the config (or take the one given) and bind the analysis.

        See the class docstring for the parameters.
        """
        super().__init__(keys_da, keys_en, sim, analysis=analysis, ensemble=ensemble)

        # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
        # are given as in the Simultaneous loop.
        self.ensemble.check_assimindex_simultaneous()

        # A single all-data-at-once update.
        self.maxiter = 1

    def check_convergence(self) -> bool:
        """ES takes a single all-data-at-once step; nothing stops early."""
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

        # Update state ensemble. This is unconditional, as it is in every other
        # scheme: the analysis result lives in enX_temp and is worthless until
        # promoted. It used to sit inside the equal-misfit branch below, which
        # is essentially never taken -- prev_data_misfit is the prior misfit and
        # data_misfit is the posterior one -- so ES returned its prior ensemble
        # unchanged while logging a reduced misfit.

        if self.data_misfit_mean == self.prev_data_misfit_mean:
            self.logger.info(
                f'ES update {self.iteration} complete!')
        else:

            # Reduction
            if self.data_misfit_mean < self.prior_data_misfit_mean:
                dF = (self.prev_data_misfit_mean - self.data_misfit_mean)/self.prev_data_misfit_mean * 100
                self.logger('ES update complete!')
                msg = f'Data Misfit reduced by {dF:.1f} %: {self.prev_data_misfit_mean:0.1f} --> {self.data_misfit_mean:0.1f}.'
                self.logger(msg)

            # Increase
            else:
                self.logger.info(
                    f'ES update complete! Objective function increased from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}.')

        self.why_stop = why_stop
        return why_stop
