"""
ES type schemes
"""
from pipt.update_schemes.enkf import enkf_approx
from pipt.update_schemes.enkf import enkf_full
from pipt.update_schemes.enkf import enkf_subspace

import numpy as np
from copy import deepcopy
from pipt.misc_tools import analysis_tools as at


class esMixIn():
    """
    This is the straightforward ES analysis scheme. We treat this as a all-data-at-once EnKF step, hence the
    calc_analysis method here is identical to that in the `enkf` class. Since, for the moment, ASSIMINDEX is parsed in a
    specific manner (or more precise, single rows and columns in the PIPT init. file is parsed to a 1D list), a
    `Simultaneous` 'loop' had to be implemented, and `es` will use this to do the inversion. Maybe in the future, we can
    make the `enkf` class do simultaneous updating also. The consequence of all this is that we inherit BOTH `enkf` and
    `Simultaneous` classes, which is convenient. The `Simultaneous` class is inherited to set up the correct inversion
    structure and `enkf` is inherited to get `calc_analysis`, so we do not have to implement it again.
    """

    def __init__(self, keys_da, keys_fwd, sim):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.

        Parameters
        ----------
        init_file : str
            PIPT init. file containing info. to run the inversion algorithm
        """
        # Pass init. file to Simultaneous parent class (Python searches parent classes from left to right).
        super().__init__(keys_da, keys_fwd, sim)

        if self.restart is False:
            # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.check_assimindex_simultaneous()

            # Extract no. assimilation steps from MDA keyword in DATAASSIM part of init. file and set this equal to
            # the number of iterations pluss one. Need one additional because the iter=0 is the prior run.
            self.max_iter = 2

    def check_convergence(self):
        """
        Calculate the "convergence" of the method. Important to
        """
        self.prev_data_misfit = self.prior_data_misfit
        # only calulate for the final (posterior) estimate
        if self.iteration == len(self.keys_da['assimindex']):
            assim_index = [self.keys_da['obsname'], list(
                np.concatenate(self.keys_da['assimindex']))]
            list_datatypes = self.list_datatypes
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)

            data_misfit = at.calc_objectivefun(
                self.full_real_obs_data, pred_data, self.full_cov_data)
            self.data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

        else:  # sequential updates not finished. Misfit is not relevant
            self.data_misfit = self.prior_data_misfit

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        if self.data_misfit == self.prev_data_misfit:
            self.logger.info(
                f'ES update {self.iteration} complete!')
            self.current_state = deepcopy(self.state)
        else:
            if self.data_misfit < self.prior_data_misfit:
                self.logger.info(
                    f'ES update complete! Objective function decreased from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
            else:
                self.logger.info(
                    f'ES update complete! Objective function increased from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        # Return conv = False, why_stop var.
        return False, True, why_stop


class es_approx(esMixIn, enkf_approx):
    """
    Mixin of ES class and approximate update
    """
    pass


class es_full(esMixIn, enkf_full):
    """
    mixin of ES class and full update.
    Note that since we do not iterate there is no difference between is full and approx.
    """
    pass


class es_subspace(esMixIn, enkf_subspace):
    """
    mixin of ES class and subspace update.
    """
    pass
