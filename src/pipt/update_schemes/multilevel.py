'''
Here we place the classes that are required to run the multilevel schemes developed in the 4DSeis project. All methods
inherit the ensemble class, hence the main loop is inherited. These classes will consider the analysis step.
'''

#──────────────────────────────────────────────────────────────────────────────────────
from pipt.loop.ensemble import Ensemble
from pipt.update_schemes.esmda import esmdaMixIn
from pipt.misc_tools import analysis_tools as at
import pipt.misc_tools.ensemble_tools as entools
from geostat.decomp import Cholesky
from pipt.update_schemes.update_methods_ns.hybrid_update import hybrid_update

import numpy as np
from copy import deepcopy
#──────────────────────────────────────────────────────────────────────────────────────


__all__ = ['multilevel', 'esmda_hybrid']

class multilevel(Ensemble):
    """
    Inititallize the multilevel class. Similar for all ML schemes, hence make one class for all.
    """
    def __init__(self, keys_da,keys_fwd,sim):
        super().__init__(keys_da, keys_fwd, sim)
        
        self.list_states = list(self.idX.keys())

        # Reorganize prior ensemble to multilevel structure if nested is true
        self.enX = self.reorganize_ml_prior(self.enX)
        self.prior_enX = deepcopy(self.enX)

        # Set ML specific options for simulator 
        self._init_sim()

        self.iteration = 0
        self.lam = 0  # set LM lamda to zero as we are doing one full update.
        if 'energy' in self.keys_da:
            self.trunc_energy = self.keys_da['energy']  # initial energy (Remember to extract this)
            if self.trunc_energy > 1:  # ensure that it is given as percentage
                self.trunc_energy /= 100.
        else:
            self.trunc_energy = 0.98

        self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
        self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(self.obs_data, self.assim_index)

        self.cov_data  = at.gen_covdata(self.datavar, self.assim_index, self.list_datatypes)
        self.vecObs, _ = at.aug_obs_pred_data(
            self.obs_data, 
            self.pred_data, 
            self.assim_index,
            self.list_datatypes
        )

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



class esmda_hybrid(multilevel,hybrid_update,esmdaMixIn):
    '''
     A multilevel implementation of the ES-MDA algorithm with the hybrid gain
    '''
    def __init__(self,keys_da, keys_fwd, sim):
        super().__init__(keys_da, keys_fwd, sim)

        self.proj = []
        for l in range(self.tot_level):
            nl = self.ml_ne[l]
            proj_l = (np.eye(nl) - np.ones((nl, nl))/nl) / np.sqrt(nl - 1)
            self.proj.append(proj_l)


    def calc_analysis(self):
        
        # Get ensemble predictions at all levels
        self.enPred = []
        for l in range(self.tot_level):
            _, enPred_level = at.aug_obs_pred_data(
                self.obs_data, 
                [el[l] for el in self.pred_data], 
                self.assim_index,
                self.list_datatypes
            )
            self.enPred.append(enPred_level) 

        # Initialize GeoStat class for generating realizations
        cholesky = Cholesky()

        if self.iteration == 1:  # first iteration

            # Note, evaluate for high fidelity model
            data_misfit = at.calc_objectivefun(
                self.enObs_conv, 
                np.concatenate(self.enPred,axis=1), # Is this correct, given the comment above??????
                self.cov_data
            )

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.prior_data_misfit_std = np.std(data_misfit)
            self.data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            # Log initial data misfit
            self.log_update(prior_run=True)
            self.data_random_state = deepcopy(np.random.get_state())


            self.ml_enObs = []
            self.scale_data = []
            self.E = []
            for l in range(self.tot_level):

                # Generate real data and scale data
                enObs_level, scale_data_level = cholesky.gen_real(
                    self.vecObs,
                    self.alpha[self.iteration - 1] * self.cov_data, 
                    self.ml_ne[l],
                    return_chol=True
                )
                self.ml_enObs.append(enObs_level)
                self.scale_data.append(scale_data_level)
                self.E.append(np.dot(enObs_level, self.proj[l]))

        else:
            self.data_random_state = deepcopy(np.random.get_state())

            for l in range(self.tot_level):
                self.ml_enObs[l], self.scale_data[l] = cholesky.gen_real(
                    self.vecObs,
                    self.alpha[self.iteration - 1] * self.cov_data,
                    self.ml_ne[l],
                    return_chol=True
                )
                self.E[l] = np.dot(self.ml_enObs[l], self.proj[l])

        # Calculate update step
        self.update(
            enX = self.enX,
            enY = self.enPred,
            enE = self.ml_enObs
        )
        if hasattr(self, 'step'):
            self.enX_temp = [self.enX[l] + self.step[l] for l in range(self.tot_level)]
            # Enforce limits
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX.keys()}
            self.enX_temp = [entools.clip_matrix(self.enX_temp[l], limits, self.idX) for l in range(self.tot_level)]

    def check_convergence(self):
        """
        Check ESMDA objective function for logging purposes.
        """

        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        enPred = []
        for l in range(self.tot_level):
            _, enPred_level = at.aug_obs_pred_data(
                self.obs_data, 
                [el[l] for el in self.pred_data], 
                self.assim_index,
                self.list_datatypes
            )
            enPred.append(enPred_level)

        data_misfit = at.calc_objectivefun(
            self.enObs_conv, 
            np.concatenate(enPred,axis=1), 
            self.cov_data
        )
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        # Log update results
        success = self.data_misfit < self.prev_data_misfit
        self.log_update(success=success)

        # Return conv = False, why_stop var.
        self.enX = deepcopy(self.enX_temp)
        self.enX_temp = None

        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        return False, True, why_stop
