"""
Build class for ensemble based optimization. This should be similar to fwd_sim/optim.py in "old" PIPT. Take what is needed.
"""

# External import
import csv  # For reading Comma Separated Values files
from importlib import import_module  # Wrapper to __import__()
import numpy as np  # Numerical tools
import os  # OS level tools
from shutil import rmtree  # for removing folders
import sys  # System level tools
from copy import deepcopy  # For dictionary copies
import pickle  # To save and load information
from glob import glob
import logging
import datetime as dt
from p_tqdm import p_map

# Internal imports
from ensemble.ensemble import Ensemble as PETEnsemble

# Internal imports
from misc.system_tools.environ_var import OpenBlasSingleThread  # Single threaded OpenBLAS runs
from pipt.misc_tools import analysis_tools as at
from popt.misc_tools import optim_tools as ot, basic_tools as bt


class Ensemble(PETEnsemble):
    """
    Class for organizing misc. variables and simulator for an optimization procedure. Here runs of the simulator and
    calculation of sensitivity matrix (if needed) is performed. In addition, methods useful in various optim_loop
    classes are/should be implemented here.
    """

    def __init__(self, keys_opt, keys_en, sim):
        """
        """
        # init PETEnsemble
        super(Ensemble, self).__init__(keys_en,sim)

        # set logger
        self.logger = logging.getLogger('PET.POPT')

        # Internalize 
        self.keys_opt = keys_opt
        self.sim = sim  # need sim class here. Restructure later



    def _load_sensitivity_info(self):
        """
        Load  information on how to calculate the sensitvity matrix.

        ST 4/5-18
        """
        # METHOD - Check which method has been choosen.
        # If SENSITIVITY contains more than one line, we need to search for METHOD
        if isinstance(self.keys_opt['sensitivity'][0], list):
            # Get indices
            ind = bt.index2d(self.keys_opt['sensitivity'], 'method')

            # Check if METHOD has been inputted
            assert None not in ind, 'METHOD not found in keyword SENSITIVITY!'

            # Assign method name
            self.sens_method = self.keys_opt['sensitivity'][ind[0]][ind[1] + 1]

        # If SENSITIVITY contains only one line, this should be METHOD
        elif isinstance(self.keys_opt['sensitivity'], list):
            # Really check if METHOD has been inputted
            assert self.keys_opt['sensitivity'] == 'method', 'METHOD not found in keyword SENSITIVITY!'

            # Assign method name
            self.sens_method = self.keys_opt['sensitivity'][1]

        # If we have ensemble sensitivity matrix, we need to load the covariance matrix from which realizations of
        # the state is made
        if self.sens_method == 'ensemble':
            # Search for COVARIANCE in SENSITIVTY
            ind_cov = bt.index2d(self.keys_opt['sensitivity'], 'cov')

            # Check if COVARIANCE has been inputted
            assert None not in ind_cov, 'You have chosen ENSEMBLE as sensitivity method but COV has not been found!'

            value_cov = self.keys_opt['sensitivity'][ind_cov[0]][ind_cov[1] + 1]
            if isinstance(value_cov, str):
                # Load covariance matrix
                load_file = np.load(value_cov)
                self.cov = load_file[load_file.files[0]]
            else:
                # Augment state
                list_state = list(self.state.keys())
                aug_state = ot.aug_optim_state(self.state, list_state)
                self.cov = value_cov * np.eye(len(aug_state))
                # Loop over the statename to assign covariance values.
                # for i, statename in enumerate(self.state.keys()):
                #     self.cov[statename] = value_cov * np.eye(self.state[statename].shape[0])

                # Search for NE (no. of ensemble members)
            ind_ne = bt.index2d(self.keys_opt['sensitivity'], 'ne')

            # Check if NE has been inputted
            assert None not in ind_ne, 'You have chosen ENSEMBLE as sensitivity method but NE has not been found!'

            # Assign NE
            self.ne = int(self.keys_opt['sensitivity'][ind_ne[0]][ind_ne[1] + 1])

    def _load_state(self):
        """
        Load the initial state for the optimization.

        ST 4/5-18
        """
        # Init. state variable
        self.state = {}

        # Make sure STATENAME and INITIALSTATE is a list so that we can loop
        if not isinstance(self.keys_opt['statename'], list):
            statename = [self.keys_opt['statename']]
            init_state = [self.keys_opt['initialstate']]
        else:
            statename = self.keys_opt['statename']
            init_state = self.keys_opt['initialstate']

        # Loop over STATENAME and assign INITIALSTATE
        for i, state in enumerate(init_state):
            # If the state is a scalar an inputted in init. file, just assign that to state
            if isinstance(state, float):
                self.state[statename[i]] = np.array([state])

            # If the state is an array, it must be loaded as a numpy save file (npz)
            elif isinstance(state, str) and state.endswith('.npz'):
                load_file = np.load(state, allow_pickle=True)
                self.state[statename[i]] = load_file[load_file.files[0]]

            # Check if a csv file has been included as "STATE". If so, we read it and make a list,
            elif isinstance(state, str) and state.endswith('.csv'):
                with open(state) as csvfile:
                    reader = csv.reader(csvfile)  # get a reader object
                    state_data = []  # Initialize the list of csv data
                    for rows in reader:  # Rows is a list of values in the csv file
                        csv_data = [None] * len(rows)
                        for col in range(len(rows)):
                            try:  # Making a float
                                csv_data[col] = float(rows[col])
                            except:  # It is a string
                                csv_data[col] = rows[col]
                        state_data.extend(csv_data)
                self.state[statename[i]] = np.array(state_data)  # Load state as an array
            else:
                print('\033[1;31mERROR: State could not be loaded!\033[1;31m')
                sys.exit(1)

    def _scale_state(self):
        for i, key in enumerate(self.state):
            self.state[key] = (self.state[key] - self.orig_lb[i]) / (self.orig_ub[i] - self.orig_lb[i])
            if np.min(self.state[key]) < 0:
                self.state[key] = 0
            elif np.max(self.state[key]) > 1:
                self.state[key] = 1

    def _invert_scale_state(self, state):
        sim_state = deepcopy(state)
        for i, key in enumerate(state):
            sim_state[key] = self.orig_lb[i] + state[key] * (self.orig_ub[i] - self.orig_lb[i])

        return sim_state

    def save_temp_state_enopt(self, ind_save):
        """
        Save the state (control variable) during an EnOPT loop. It is stored in a list with length equal to max.
        iteration length + 1 (initial state is saved in entry 0). The list of temporary states are stored in a npz
        file.
        OBS: Max. iterations must be defined before invoking this method.

        Input:
                - state:            Dict. of state
                - ind_save:         Iteration step to save (0 = init. state)

        Output:
                                    Temporary snapshot of state saved as Numpy save file

        ST 4/5-18: Copied most of the code from fwd_sim.ensemble.save_temp_state_iter.
        """
        # Initial save
        if ind_save == 0:
            self.temp_state = deepcopy(self.state)
            self.temp_obj_func = []
            self.temp_obj_func_en = []
            self.temp_alpha = [self.keys_opt['enopt'][3][1]]
            self.temp_cov = [self.keys_opt['sensitivity'][2][1]]

        else:
            for key in self.state:
                self.temp_state[key] = np.vstack((self.temp_state[key], self.state[key]))
            self.temp_obj_func.append(self.obj_func)
            self.temp_obj_func_en.append(self.obj_func_en)
            self.temp_alpha.append(self.alpha)
            self.temp_cov.append(self.cov[0, 0])

        np.savez('temp_state_enopt', state=self.temp_state, obj_func=self.temp_obj_func,
                 obj_func_en=self.temp_obj_func_en, alpha=self.temp_alpha, cov=self.temp_cov)

    def get_obj_func(self, pred_data_en):
        if self.keys_opt['objfunc'][0] == 'npv':
            # Calculate initial estimated economic value for each time period
            # report dates are yearly here.
            npv = []
            for member in range(len(pred_data_en)):
                pred_data = pred_data_en[member]
                initial_gh = []
                for i in np.arange(1, len(pred_data)):
                    if i == 1:
                        # test_result = self. ecl.get_sim_results('WOPR PRO-1', ['dates', self.report[i]])
                        Qop = pred_data[i]['FOPT']
                        Qgp = pred_data[i]['FGPT']
                        Qwp = pred_data[i]['FWPT']
                        Qwi = pred_data[i]['FWIT']
                        delta_days = self.sim.report['days'][i - 1]
                    else:
                        Qop = pred_data[i]['FOPT'] - pred_data[i - 1]['FOPT']
                        Qgp = pred_data[i]['FGPT'] - pred_data[i - 1]['FGPT']
                        Qwp = pred_data[i]['FWPT'] - pred_data[i - 1]['FWPT']
                        Qwi = pred_data[i]['FWIT'] - pred_data[i - 1]['FWIT']
                        delta_days = self.sim.report['days'][i] - self.sim.report['days'][i - 1]

                    wop = self.keys_opt['npv_const'][0][1]
                    wgp = self.keys_opt['npv_const'][1][1]
                    wwp = self.keys_opt['npv_const'][2][1]
                    wwi = self.keys_opt['npv_const'][3][1]
                    disc = self.keys_opt['npv_const'][4][1]

                    val = (Qop * wop + Qgp * wgp - Qwp * wwp - Qwi * wwi) / (
                            (1 + disc) ** (delta_days / 365))
                    initial_gh = np.append(initial_gh, val)

                initial_gh = [float("%.6f" % it) for it in initial_gh]
                npv.append(sum(initial_gh) / self.keys_opt['npv_const'][5][1])

            return npv

    def calc_ensemble_sensitivity(self):
        """
        Calculate the sensitivity matrix normally associated with ensemble optimization algorithms, usually defined as:

            S ~= C_x * G.T

        where '~=' means 'approximately equal', C_x is the state covariance matrix, and G is the standard
        sensitivity matrix. The ensemble sensitivity matrix is calculated as:

            S = (1/ (ne - 1)) * U * J.T

        where U and J are ensemble matrices of state (or control variables) and objective function perturbed by their
        respective means. In practice (and in this method), S is calculated by perturbing the current state (control
        variable) with Gaussian random numbers from N(0, C_x) (giving U), running the generated state ensemble (U)
        through the simulator to give an ensemble of objective function values (J), and in the end calculate S. Note
        that S is an Ns x 1 vector, where Ns is length of the state vector (the objective function is just a scalar!)

        ST 3/5-18: First implementation. Much of the code here is taken directly from fwd_sim.ensemble Ensemble class.
        YC 2/10-19: Added calcuating gradient of covariance.
        """
        # Generate ensemble of states
        state_en = self._gen_state_ensemble()

        self.run_ensemble(state_en)

        # Finally, we calculate the ensemble sensitivity matrix.
        # First we need to perturb state and obj. func. ensemble with their mean. Note that, obj_func has shape (
        # ne,)!
        list_states = list(self.state.keys())
        aug_state = at.aug_state(state_en, list_states)
        pert_state = aug_state - np.dot(aug_state.mean(1)[:, None], np.ones((1, self.ne)))
        pert_obj_func = self.obj_func_en - np.tile(np.mean(self.obj_func_en), (1, self.ne))

        # Calculate cross-covariance between state and obj. func. which is the ensemble sensitivity matrix
        self.sens_matrix = at.calc_crosscov(pert_state, pert_obj_func)

        # Calculate the gradient for covariance matrix
        g_c = np.zeros(self.cov.shape)
        for i in np.arange(self.ne):
            state_tmp = aug_state[:, i] - aug_state.mean(1)
            g_c = g_c + pert_obj_func[0, i] * (np.outer(state_tmp, state_tmp) - self.cov)

        self.cov_sens_matrix = g_c / self.ne

    def run_ensemble(self, state):
        """
        Calculate the forward simulation response of the current state of the optimization procedure. The procedure
        for calculating forward response is:

            i. Setup and run simulator with current state
            ii. Get result from ended simulation

        The procedure here is general, hence a simulator used here must contain the initial step of setting up the
        parameters and steps i-ii. Initialization of the simulator is done when initializing the Optim class (see
        __init__()). The names of the mandatory methods in a simulator are:

            > setup_fwd_sim
            > run_fwd_sim
            > get_obj_func


        ST 25/9-19: First implementation.
        """
        # Initialize variables for the while-loop

        if len(self.keys_opt['objfunc']) > 1 and self.keys_opt['objfunc'][1] > 1:
            nm = self.keys_opt['objfunc'][1]  # number of models (e.g., geo-models)
        else:
            nm = 1
        kl = list(state.keys())
        ns = 1
        if len(state[kl[0]].shape) > 1:
            ns = state[kl[0]].shape[1]  # number of control vectors in state (1 when evaluating mean control, and N when evaluating gradient)
        ne = np.maximum(nm, ns)

        no_tot_run = int(self.sim.input_dict['parallel'])  # How many to run in parallel
        self.obj_func_en = np.zeros(ne)  # Init. obj. func. output (responses from simulator)

        report_ind = ['days', list(range(len(self.sim.reportdates) - 1))]
        #self._init_pred_data(report_ind, ne)  # Initialize pred_data in the simulator
        l_prim = [int(x) for x in report_ind[1]]
        true_list = []
        for i in l_prim:
            true_list.append((self.sim.reportdates[i + 1] - self.sim.reportdates[0]).days)
        true_days = ['days', true_list]

        self.sim.setup_fwd_run(report_ind, true_days, l_prim, self.pred_data)

        # scale back
        sim_state = self._invert_scale_state(state)

        # Here we copy the control perturbations to list_state.
        # If only the mean is run, we duplicate it to number of models
        list_state = [deepcopy({}) for _ in range(ne)]
        for i in range(ne):
            for key in sim_state.keys():
                if ns == 1:
                    list_state[i][key] = deepcopy(sim_state[key])
                else:
                    list_state[i][key] = deepcopy(sim_state[key][:, i])
        list_member_index = [i for i in range(ne)]

        en_pred = p_map(self.sim.run_fwd_sim, list_state, list_member_index, num_cpus=no_tot_run)

        self.obj_func_en = self.get_obj_func(en_pred)

        # replace failed runs
        list_crash = [indx for indx, el in enumerate(en_pred) if el is False]
        list_success = [indx for indx, el in enumerate(en_pred) if el is not False]

        if not len(list_success):  # all runs have crashed
            self.save()
            print('\n\033[1;31mERROR: All started simulations has failed! We dump all information and exit!\033[1;m')
            sys.exit(1)

        if len(list_crash):
            if len(list_crash) < len(list_success):  # more successfull than crashed runs
                copy_member = np.random.choice(list_success, size=len(list_crash), replace=False)
            else:
                copy_member = np.random.choice(list_success, size=len(list_crash), replace=True)

            for indx, el in enumerate(copy_member):
                print(
                    f'\033[92m--- Ensemble member {list_crash[indx]} failed, has been replaced by ensemble member {el}! ---\033[92m')
                for key in state.keys():
                    state[key][:, list_crash[indx]] = deepcopy(state[key][:, el])
                en_pred[list_crash[indx]] = deepcopy(en_pred[el])

    def _gen_inputs(self):
        """
        Generate state inputs to collect prediction data for ML.
        YC 9/10-19
        """
        state_en = {}
        if 'scaling' in self.keys_opt:
            # TODO: make the bounds more general
            scaling_lowerBound, scaling_upperBound = self.keys_opt['scaling'][0], self.keys_opt['scaling'][1]

        for i, statename in enumerate(self.state.keys()):
            temp_state_en = np.random.uniform(low=scaling_lowerBound, high=scaling_upperBound,
                                              size=(self.state[statename].size, self.ne))

            state_en[statename] = temp_state_en

        return state_en

    def _gen_state_ensemble(self):
        """
        Generate an ensemble of states (control variables) to run in calc_ensemble_sensitivity. It is assumed that
        the covariance function needed to generate realizations has been inputted via the SENSITIVITY keyword (with
        METHOD option ENSEMBLE).

        ST 4/5-18
        """
        # TODO: Gen. realizations for control variables at separate time steps (cov is a block diagonal matrix),
        # and for more than one STATENAME

        # # Initialize Cholesky class
        # chol = decomp.Cholesky()
        #
        # # Augment state
        # list_state = list(self.state.keys())
        # aug_state = ot.aug_optim_state(self.state, list_state)

        # Generate ensemble with the current state (control variable) as the mean and using the imported covariance
        # matrix
        state_en = {}
        for i, statename in enumerate(self.state.keys()):
            # state_en[statename] = chol.gen_real(self.state[statename], self.cov[i, i], self.ne)
            mean = self.state[statename]
            len_state = len(self.state[statename])
            cov = self.cov[len_state * i:len_state * (i + 1), len_state * i:len_state * (i + 1)]
            if len(cov) != len(mean):  # make sure cov is diagonal matrix
                print('\033[1;31mERROR: Covariance must be diagonal matrix!\033[1;31m')
            #     cov = cov*np.identity(len(mean))
            temp_state_en = np.random.multivariate_normal(mean, cov, self.ne).transpose()
            if 'scaling' in self.keys_opt:
                # TODO: make the bounds more general
                scaling_lowerBound, scaling_upperBound = self.keys_opt['scaling'][0], self.keys_opt['scaling'][1]
                np.clip(temp_state_en, scaling_lowerBound, scaling_upperBound, out=temp_state_en)

            state_en[statename] = temp_state_en

        return state_en

    def save(self):
        """
        We use pickle to dump all the information we have in 'self'. Can be used, e.g., if some error has occurred.

        ST 28/2-17
        """
        # Open save file and dump all info. in self
        with open(self.pickle_restart_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        """
        Load a pickled file and save all info. in self.

        ST 28/2-17
        """
        # Open file and read with pickle
        with open(self.pickle_restart_file, 'rb') as f:
            tmp_load = pickle.load(f)

        # Save in 'self'
        self.__dict__.update(tmp_load)
