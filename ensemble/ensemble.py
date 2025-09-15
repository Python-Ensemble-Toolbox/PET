"""
Package contains the basis for the PET ensemble based structure.
"""

# External imports
import csv  # For reading Comma Separated Values files
import os  # OS level tools
import sys  # System-specific parameters and functions
from copy import deepcopy, copy  # Copy functions. (deepcopy let us copy mutable items)
from shutil import rmtree  # rmtree for removing folders
import numpy as np  # Misc. numerical tools
import pickle  # To save and load information
from glob import glob
import datetime as dt
from tqdm.auto import tqdm
from p_tqdm import p_map
import logging

# Internal imports
import pipt.misc_tools.analysis_tools as at
from geostat.decomp import Cholesky  # Making realizations
from pipt.misc_tools import cov_regularization
from pipt.misc_tools import wavelet_tools as wt
from misc import read_input_csv as rcsv
from misc.system_tools.environ_var import OpenBlasSingleThread  # Single threaded OpenBLAS runs


class Ensemble:
    """
    Class for organizing misc. variables and simulator for an ensemble-based inversion run. Here, the forecast step
    and prediction runs are performed. General methods that are useful in various ensemble loops have also been
    implemented here.
    """

    def __init__(self, keys_en, sim, redund_sim=None):
        """
        Class extends the ReadInitFile class. First the PIPT init. file is passed to the parent class for reading and
        parsing. Rest of the initialization uses the keywords parsed in ReadInitFile (parent) class to set up observed,
        predicted data and data variance dictionaries. Also, the simulator to be used in forecast and/or predictions is
        initialized with keywords parsed in ReadInitFile (parent) class. Lastly, the initial ensemble is generated (if
        it has not been inputted), and some saving of variables can be done chosen in PIPT init. file.

        Parameter
        ---------
        init_file : str
                    path to input file containing initiallization values
        """
        # Internalize PET dictionary
        self.keys_en = keys_en
        self.sim = sim
        self.sim.redund_sim = redund_sim
        self.pred_data = None

        # Auxilliary input to the simulator - can be used e.g.,
        # to allow for different models when optimizing.
        self.aux_input = None

        # Setup logger
        logging.basicConfig(level=logging.INFO,
                            filename='pet_logger.log',
                            filemode='w',
                            format='%(asctime)s : %(levelname)s : %(name)s : %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('PET')

        # Check if folder contains any En_ files, and remove them!
        for folder in glob('En_*'):
            try:
                if len(folder.split('_')) == 2:
                    int(folder.split('_')[1])
                    rmtree(folder)
            except:
                pass

        # Save name for (potential) pickle dump/load
        self.pickle_restart_file = 'emergency_dump'

        # initiallize the restart. Standard is no restart
        self.restart = False

        # If it is a restart run, we do not need to initialize anything, only load the self info. that exists in the
        # pickle save file. If it is not a restart run, we initialize everything below.
        if ('restart' in self.keys_en) and (self.keys_en['restart'] == 'yes'):
            # Initiate a restart run
            self.logger.info('\033[92m--- Restart run initiated! ---\033[92m')
            # Check if the pickle save file exists in folder
            try:
                assert (self.pickle_restart_file in [
                        f for f in os.listdir('.') if os.path.isfile(f)])
            except AssertionError as err:
                self.logger.exception('The restart file "{0}" does not exist in folder. Cannot restart!'.format(
                    self.pickle_restart_file))
                raise err

            # Load restart file
            self.load()

            # Ensure that restart switch is ON since the error may not have happened during a restart run
            self.restart = True

        # Init. various variables/lists/dicts. needed in ensemble run
        else:
            # delete potential restart files to avoid any problems
            if self.pickle_restart_file in [f for f in os.listdir('.') if os.path.isfile(f)]:
                os.remove(self.pickle_restart_file)

            # initialize sim limit
            if 'sim_limit' in self.keys_en:
                self.sim_limit = self.keys_en['sim_limit']
            else:
                self.sim_limit = float('inf')

            # bool that can be used to supress tqdm output (useful when testing code)
            if 'disable_tqdm' in self.keys_en:
                self.disable_tqdm = self.keys_en['disable_tqdm']
            else:
                self.disable_tqdm = False

            # extract information that is given for the prior model
            self.prior_info = self._extract_prior_info()

            # Calculate initial ensemble if IMPORTSTATICVAR has not been given in init. file.
            # Prior info. on state variables must be given by PRIOR_<STATICVAR-name> keyword.
            if 'importstaticvar' not in self.keys_en:
                self.ne = int(self.keys_en['ne'])

                # Output = self.state, self.cov_prior
                self.gen_init_ensemble()

            else:
                # State variable imported as a Numpy save file
                tmp_load = np.load(self.keys_en['importstaticvar'], allow_pickle=True)

                # We assume that the user has saved the state dict. as **state (effectively saved all keys in state
                # individually).
                self.state = {key: val for key, val in tmp_load.items()}

                # Find the number of ensemble members from state variable
                tmp_ne = []
                for tmp_state in self.state.keys():
                    tmp_ne.extend([self.state[tmp_state].shape[1]])
                if max(tmp_ne) != min(tmp_ne):
                    print('\033[1;33mInput states have different ensemble size\033[1;m')
                    sys.exit(1)
                self.ne = min(tmp_ne)
        self._ext_ml_info()

    def _ext_ml_info(self):
        '''
        Extract the info needed for ML simulations. Note if the ML keyword is not in keys_en we initialize
        such that we only have one level -- the high fidelity one
        '''

        if 'multilevel' in self.keys_en:
            # parse
            self.multilevel = {}
            self.ML_error_corr = 'none'
            for i, opt in enumerate(list(zip(*self.keys_en['multilevel']))[0]):
                if opt == 'levels':
                    self.multilevel['levels'] = [elem for elem in range(
                        int(self.keys_en['multilevel'][i][1]))]
                    self.tot_level = int(self.keys_en['multilevel'][i][1])
                if opt == 'en_size':
                    self.multilevel['ne'] = [range(int(el))
                                             for el in self.keys_en['multilevel'][i][1]]
                    self.ml_ne = [int(el) for el in self.keys_en['multilevel'][i][1]]
                if opt == 'ml_error_corr':
                    # options for ML_error_corr are: bias_corr, deterministic, stochastic, telescopic
                    self.ML_error_corr = self.keys_en['multilevel'][i][1]
                    if not self.ML_error_corr == 'none':
                        # options for error_comp_scheme are: once, ens, sep
                        self.error_comp_scheme = self.keys_en['multilevel'][i][2]
                    self.ML_corr_done = False

    def _extract_prior_info(self) -> dict:
        '''
        Extract prior information on STATE from keyword(s) PRIOR_<STATE entries>.
        '''

        # Get state names as list
        state_names = self.keys_en['state']
        if not isinstance(state_names, list): state_names = [state_names]

        # Check if PRIOR_<state names> exists for each entry in state
        for name in state_names:
            assert f'prior_{name}' in self.keys_en, \
                'PRIOR_{0} is missing! This keyword is needed to make initial ensemble for {0} entered in ' \
                'STATE'.format(name.upper())
        
        # define dict to store prior information in 
        prior_info = {name: None for name in state_names}

        # loop over state priors
        for name in state_names:
            prior = self.keys_en[f'prior_{name}']
            
            # Check if is a list (old way)
            if isinstance(prior, list):
                # list of lists - old way of inputting prior information
                prior_dict = {}
                for i, opt in enumerate(list(zip(*prior))[0]):
                    if opt == 'limits':
                        prior_dict[opt] = prior[i][1:]
                    else:
                        prior_dict[opt] = prior[i][1]
                prior = prior_dict
            else:
                assert isinstance(prior, dict), 'PRIOR_{0} must be a dictionary or list of lists!'.format(name.upper())


            # load mean if in file
            if isinstance(prior['mean'], str):
                assert prior['mean'].endswith('.npz'), 'File name does not end with \'.npz\'!'
                load_file = np.load(prior['mean'])
                assert len(load_file.files) == 1, \
                    'More than one variable located in {0}. Only the mean vector can be stored in the .npz file!' \
                    .format(prior['mean'])
                prior['mean'] = load_file[load_file.files[0]]
            else:  # Single number inputted, make it a list if not already
                if not isinstance(prior['mean'], list):
                    prior['mean'] = [prior['mean']]

            # loop over keys in prior
            for key in prior.keys():
                # ensure that entry is a list
                if (not isinstance(prior[key], list)) and (key != 'mean'):
                    prior[key] = [prior[key]]

            # change the name of some keys
            prior['variance'] = prior.pop('var', None)
            prior['corr_length'] = prior.pop('range', None)

            # process grid
            if 'grid' in prior:
                grid_dim = prior['grid']

                # check if 3D-grid
                if (len(grid_dim) == 3) and (grid_dim[2] > 1):
                    nz = int(grid_dim[2])
                    prior['nz'] = nz
                    prior['nx'] = int(grid_dim[0])
                    prior['ny'] = int(grid_dim[1])
                    

                    # Check mean when values have been inputted directly (not when mean has been loaded)
                    mean = prior['mean']
                    if isinstance(mean, list) and len(mean) < nz:
                         # Check if it is more than one entry and give error
                        assert len(mean) == 1, \
                            'Information from MEAN has been given for {0} layers, whereas {1} is needed!' \
                            .format(len(mean), nz)

                        # Only 1 entry; copy this to all layers
                        print(
                            '\033[1;33mSingle entry for MEAN will be copied to all {0} layers\033[1;m'.format(nz))
                        prior['mean'] = mean * nz

                    #check if info. has been given on all layers. In the case it has not been given, we just copy the info. given.
                    for key in ['vario', 'variance', 'aniso', 'angle', 'corr_length']:
                        if key in prior.keys():
                            val = prior[key]
                            if len(val) < nz:
                                # Check if it is more than one entry and give error
                                assert len(val) == 1, \
                                    'Information from {0} has been given for {1} layers, whereas {2} is needed!' \
                                    .format(key.upper(), len(val), nz)

                                # Only 1 entry; copy this to all layers
                                print(
                                    '\033[1;33mSingle entry for {0} will be copied to all {1} layers\033[1;m'.format(key.upper(), nz))
                                prior[key] = val * nz

                else:
                    prior['nx'] = int(grid_dim[0])
                    prior['ny'] = int(grid_dim[1])
                    prior['nz'] = 1

            prior.pop('grid', None)

            # add prior to prior_info
            prior_info[name] = prior
            
        return prior_info
                

    def gen_init_ensemble(self):
        """
        Generate the initial ensemble of (joint) state vectors using the GeoStat class in the "geostat" package.
        TODO: Merge this function with the perturbation function _gen_state_ensemble in popt.
        """
        # Initialize GeoStat class
        init_en = Cholesky()

        # (Re)initialize state variable as dictionary
        self.state = {}
        self.cov_prior = {}

        for name in self.prior_info:
            # Init. indices to pick out correct mean vector for each layer
            ind_end = 0

            # Extract info.
            nx = self.prior_info[name].get('nx', 0)
            ny = self.prior_info[name].get('ny', 0)
            nz = self.prior_info[name].get('nz', 0)
            mean = self.prior_info[name].get('mean', None)

            if nx == ny == 0:  # assume ensemble will be generated elsewhere if dimensions are zero
                break

            variance = self.prior_info[name].get('variance', None)
            corr_length = self.prior_info[name].get('corr_length', None)
            aniso = self.prior_info[name].get('aniso', None)
            vario = self.prior_info[name].get('vario', None)
            angle = self.prior_info[name].get('angle', None)
            limits= self.prior_info[name].get('limits',None)
            

            # Loop over nz to make layers of 2D priors
            for i in range(self.prior_info[name]['nz']):
                # If mean is scalar, no covariance matrix is needed
                if type(self.prior_info[name]['mean']).__module__ == 'numpy':
                    # Generate covariance matrix
                    cov = init_en.gen_cov2d(
                        nx, ny, variance[i], corr_length[i], aniso[i], angle[i], vario[i])
                else:
                    cov = np.array(variance[i])

                # Pick out the mean vector for the current layer
                ind_start = ind_end
                ind_end = int((i + 1) * (len(mean) / nz))
                mean_layer = mean[ind_start:ind_end]

                # Generate realizations. If LIMITS have been entered, they must be taken account for here
                if limits is None:
                    real = init_en.gen_real(mean_layer, cov, self.ne)
                else:
                    real = init_en.gen_real(mean_layer, cov, self.ne, {
                                            'upper': limits[i][1], 'lower': limits[i][0]})

                # Stack realizations for each layer
                if i == 0:
                    real_out = real
                else:
                    real_out = np.vstack((real_out, real))

            # Store realizations in dictionary with name given in STATICVAR
            self.state[name] = real_out

            # Store the covariance matrix
            self.cov_prior[name] = cov
        
        # Save the ensemble for later inspection
        np.savez('prior.npz', **self.state)

    def get_list_assim_steps(self):
        """
        Returns list of assimilation steps. Useful in a 'loop'-script.

        Returns
        -------
        list_assim : list
                     List of total assimilation steps.
        """
        # Get list of assim. steps. from ASSIMINDEX
        list_assim = list(range(len(self.keys_da['assimindex'])))

        # If it is a restart run, we only list the assimilation steps we have not done
        if self.restart is True:
            # List simulations we already have done. Do this by checking pred_data.
            # OBS: Minus 1 here do to the aborted simulation is also not None.
            sim_done = list(
                range(len([ind for ind, p in enumerate(self.pred_data) if p is not None]) - 1))

            # Update list of assim. steps by removing simulations we have done
            list_assim = [ind for ind in list_assim if ind not in sim_done]

        # Return tot. assim. steps
        return list_assim

    def calc_prediction(self, input_state=None, save_prediction=None):
        """
        Method for making predictions using the state variable. Will output the simulator response for all report steps
        and all data values provided to the simulator.

        Parameters
        ----------
        input_state :
            Use an input state instead of internal state (stored in self) to run predictions
        save_prediction :
            Save the predictions as a <save_prediction>.npz file (numpy compressed file)

        Returns
        -------
        prediction :
            List of dictionaries with keys equal to data types (in DATATYPE),
            containing the responses at each time step given in PREDICTION.

        """

        if isinstance(self.state,list) and hasattr(self, 'multilevel'): # assume multilevel is used if state is a list 
            success = self.calc_ml_prediction(input_state)
        else:
            # Number of parallel runs
            if 'parallel' in self.sim.input_dict:
                no_tot_run = int(self.sim.input_dict['parallel'])
            else:
                no_tot_run = 1
            self.pred_data = []

            # for level in self.multilevel['level']: #
            # Setup forward simulator and redundant simulator at the correct fidelity
            if self.sim.redund_sim is not None:
                self.sim.redund_sim.setup_fwd_run()
            self.sim.setup_fwd_run(redund_sim=self.sim.redund_sim)

            # Ensure that we put all the states in a list
            list_state = [deepcopy({}) for _ in range(self.ne)]
            for i in range(self.ne):
                if input_state is None:
                    for key in self.state.keys():
                        if self.state[key].ndim == 1:
                            list_state[i][key] = deepcopy(self.state[key])
                        elif self.state[key].ndim == 2:
                            list_state[i][key] = deepcopy(self.state[key][:, i])
                        # elif self.state[key].ndim == 3:
                        #     list_state[i][key] = deepcopy(self.state[key][level,:, i])
                else:
                    for key in self.state.keys():
                        if input_state[key].ndim == 1:
                            list_state[i][key] = deepcopy(input_state[key])
                        elif input_state[key].ndim == 2:
                            list_state[i][key] = deepcopy(input_state[key][:, i])
                        # elif input_state[key].ndim == 3:
                        #     list_state[i][key] = deepcopy(input_state[key][:,:, i])
                if self.aux_input is not None:  # several models are used
                    list_state[i]['aux_input'] = self.aux_input[i]

            # Index list of ensemble members
            list_member_index = list(range(self.ne))

            if no_tot_run==1: # if not in parallel we use regular loop
                en_pred = [self.sim.run_fwd_sim(state, member_index) for state, member_index in
                           tqdm(zip(list_state, list_member_index), total=len(list_state))]
            elif self.sim.input_dict.get('hpc', False): # Run prediction in parallel on hpc
                batch_size = no_tot_run # If more than 500 ensemble members, we limit the runs to batches of 500
                # Split the ensemble into batches of 500
                if batch_size >= 1000:
                    self.logger.info(f'Cannot run batch size of {no_tot_run}. Set to 1000')
                    batch_size = 1000
                en_pred = []
                batch_en = [np.arange(start, start + batch_size) for start in
                            np.arange(0, self.ne - batch_size, batch_size)]
                if len(batch_en): # if self.ne is less than batch_size
                    batch_en.append(np.arange(batch_en[-1][-1]+1, self.ne))
                else:
                    batch_en.append(np.arange(0, self.ne))
                for n_e in batch_en:
                    _ = [self.sim.run_fwd_sim(state, member_index, nosim=True) for state, member_index in
                            zip([list_state[curr_n] for curr_n in n_e], [list_member_index[curr_n] for curr_n in n_e])]
                    # Run call_sim on the hpc
                    if self.sim.options['mpiarray']:
                        job_id = self.sim.SLURM_ARRAY_HPC_run(
                                                            n_e,
                                                            venv=os.path.join(os.path.dirname(sys.executable), 'activate'),
                                                            filename=self.sim.file,
                                                            **self.sim.options
                                                        )
                    else:
                        job_id=self.sim.SLURM_HPC_run(
                                                    n_e, 
                                                    venv=os.path.join(os.path.dirname(sys.executable),'activate'),
                                                    filename=self.sim.file,
                                                    **self.sim.options
                                                    )
                    
                    # Wait for the simulations to finish
                    if job_id:
                        sim_status = self.sim.wait_for_jobs(job_id)
                    else:
                        print("Job submission failed. Exiting.")
                        sim_status = [False]*len(n_e)
                    # Extract the results. Need a local counter to check the results in the correct order
                    for c_member, member_i in enumerate([list_member_index[curr_n] for curr_n in n_e]):
                        if sim_status[c_member]:
                            self.sim.extract_data(member_i)
                            en_pred.append(deepcopy(self.sim.pred_data))
                            if self.sim.saveinfo is not None:  # Try to save information
                                store_ensemble_sim_information(self.sim.saveinfo, member_i)
                        else:
                            en_pred.append(False)
                        self.sim.remove_folder(member_i)
            else: # Run prediction in parallel using p_map
                en_pred = p_map(self.sim.run_fwd_sim, list_state,
                                list_member_index, num_cpus=no_tot_run, disable=self.disable_tqdm)
            # List successful runs and crashes
            list_crash = [indx for indx, el in enumerate(en_pred) if el is False]
            list_success = [indx for indx, el in enumerate(en_pred) if el is not False]
            success = True

            # Dump all information and print error if all runs have crashed
            if not list_success:
                self.save()
                success = False
                if len(list_crash) > 1:
                    print(
                        '\n\033[1;31mERROR: All started simulations has failed! We dump all information and exit!\033[1;m')
                    self.logger.info(
                        '\n\033[1;31mERROR: All started simulations has failed! We dump all information and exit!\033[1;m')
                    sys.exit(1)
                return success

            # Check crashed runs
            if list_crash:
                # Replace crashed runs with (random) successful runs. If there are more crashed runs than successful once,
                # we draw with replacement.
                if len(list_crash) < len(list_success):
                    copy_member = np.random.choice(
                        list_success, size=len(list_crash), replace=False)
                else:
                    copy_member = np.random.choice(
                        list_success, size=len(list_crash), replace=True)

                # Insert the replaced runs in prediction list
                for indx, el in enumerate(copy_member):
                    print(f'\033[92m--- Ensemble member {list_crash[indx]} failed, has been replaced by ensemble member '
                          f'{el}! ---\033[92m')
                    self.logger.info(f'\033[92m--- Ensemble member {list_crash[indx]} failed, has been replaced by '
                                     f'ensemble member {el}! ---\033[92m')
                    for key in self.state.keys():
                        if self.state[key].ndim > 1:
                            self.state[key][:, list_crash[indx]] = deepcopy(
                                self.state[key][:, el])
                    en_pred[list_crash[indx]] = deepcopy(en_pred[el])
 
            # Convert ensemble specific result into pred_data, and filter for NONE data
            self.pred_data.extend([{typ: np.concatenate(tuple((el[ind][typ][:, np.newaxis]) for el in en_pred), axis=1)
                                    if any(elem is not None for elem in tuple((el[ind][typ]) for el in en_pred))
                                    else None for typ in en_pred[0][0].keys()} for ind in range(len(en_pred[0]))])

        # some predicted data might need to be adjusted (e.g. scaled or compressed if it is 4D seis data). Do not
        # include this here.

        # Store results if needed
        if save_prediction is not None:
            np.savez(f'{save_prediction}.npz', **{'pred_data': self.pred_data})

        return success

    def save(self):
        """
        We use pickle to dump all the information we have in 'self'. Can be used, e.g., if some error has occurred.

        Changelog
        ---------
        - ST 28/2-17
        """
        # Open save file and dump all info. in self
        with open(self.pickle_restart_file, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=4)

    def load(self):
        """
        Load a pickled file and save all info. in self.

        Changelog
        ---------
        - ST 28/2-17
        """
        # Open file and read with pickle
        with open(self.pickle_restart_file, 'rb') as f:
            tmp_load = pickle.load(f)

        # Save in 'self'
        self.__dict__.update(tmp_load)

    def calc_ml_prediction(self, input_state=None):
        """
        Function for running the simulator over several levels. We assume that it is sufficient to provide the level
        integer to the setup of the forward run. This will initiate the correct simulator fidelity.
        The function then runs the set of state through the different simulator fidelities.

        Parameters
        ----------
        input_state:
            If simulation is run stand-alone one can input any state.
        """

        no_tot_run = int(self.sim.input_dict['parallel'])
        ml_pred_data = []

        for level in tqdm(self.multilevel['levels'], desc='Fidelity level', position=1):
            # Setup forward simulator and redundant simulator at the correct fidelity
            if self.sim.redund_sim is not None:
                self.sim.redund_sim.setup_fwd_run(level=level)
            self.sim.setup_fwd_run(level=level)
            ml_ne = self.multilevel['ne'][level]
            if ml_ne:
                # Ensure that we put all the states in a list
                list_state = [deepcopy({}) for _ in ml_ne]
                for i in ml_ne:
                    if input_state is None:
                        for key in self.state[level].keys():
                            if self.state[level][key].ndim == 1:
                                list_state[i][key] = deepcopy(self.state[level][key])
                            elif self.state[level][key].ndim == 2:
                                list_state[i][key] = deepcopy(self.state[level][key][:, i])
                    else:
                        for key in self.state.keys():
                            if input_state[level][key].ndim == 1:
                                list_state[i][key] = deepcopy(input_state[level][key])
                            elif input_state[level][key].ndim == 2:
                                list_state[i][key] = deepcopy(input_state[level][key][:, i])
                    if self.aux_input is not None:  # several models are used
                        list_state[i]['aux_input'] = self.aux_input[i]

                # Index list of ensemble members
                list_member_index = list(ml_ne)

                # Run prediction in parallel using p_map
                en_pred = p_map(self.sim.run_fwd_sim, list_state,
                                list_member_index, num_cpus=no_tot_run, disable=self.disable_tqdm)

                # List successful runs and crashes
                list_crash = [indx for indx, el in enumerate(en_pred) if el is False]
                list_success = [indx for indx, el in enumerate(en_pred) if el is not False]
                success = True

                # Dump all information and print error if all runs have crashed
                if not list_success:
                    self.save()
                    success = False
                    if len(list_crash) > 1:
                        print(
                            '\n\033[1;31mERROR: All started simulations has failed! We dump all information and exit!\033[1;m')
                        self.logger.info(
                            '\n\033[1;31mERROR: All started simulations has failed! We dump all information and exit!\033[1;m')
                        sys.exit(1)
                    return success

                # Check crashed runs
                if list_crash:
                    # Replace crashed runs with (random) successful runs. If there are more crashed runs than successful once,
                    # we draw with replacement.
                    if len(list_crash) < len(list_success):
                        copy_member = np.random.choice(
                            list_success, size=len(list_crash), replace=False)
                    else:
                        copy_member = np.random.choice(
                            list_success, size=len(list_crash), replace=True)

                    # Insert the replaced runs in prediction list
                    for indx, el in enumerate(copy_member):
                        print(f'\033[92m--- Ensemble member {list_crash[indx]} failed, has been replaced by ensemble member '
                              f'{el}! ---\033[92m')
                        self.logger.info(f'\033[92m--- Ensemble member {list_crash[indx]} failed, has been replaced by '
                                         f'ensemble member {el}! ---\033[92m')
                        for key in self.state[level].keys():
                            self.state[level][key][:, list_crash[indx]] = deepcopy(
                                self.state[level][key][:, el])
                        en_pred[list_crash[indx]] = deepcopy(en_pred[el])

                # Convert ensemble specific result into pred_data, and filter for NONE data
                ml_pred_data.append([{typ: np.concatenate(tuple((el[ind][typ][:, np.newaxis]) for el in en_pred), axis=1)
                                      if any(elem is not None for elem in tuple((el[ind][typ]) for el in en_pred))
                                      else None for typ in en_pred[0][0].keys()} for ind in range(len(en_pred[0]))])

        # loop over time instance first, and the level instance.
        self.pred_data = np.array(ml_pred_data).T.tolist()

        if hasattr(self,'treat_modeling_error'):
            self.treat_modeling_error()

        return success

    def treat_modeling_error(self):
        if not self.ML_error_corr=='none':
            if self.error_comp_scheme=='sep':
                self.calc_modeling_error_sep()
                self.address_ML_error()
            elif self.error_comp_scheme=='once':
                if not self.ML_corr_done:
                    self.calc_modeling_error_ens()
                    self.ML_corr_done = True
                self.address_ML_error()
            elif self.error_comp_scheme=='ens':
                self.calc_modeling_error_ens()

    def calc_modeling_error_sep(self):
        print('calc_modeling_error_sep -- Not yet implemented')

    def calc_modeling_error_ens(self):

        if self.ML_error_corr =='bias_corr':
            # modify self.pred_data without changing its structure. Hence, for each level (except the finest one)
            # we correct each data at each point in time.
            for assim_index in range(len(self.pred_data)):
                for dat in self.pred_data[assim_index][-1].keys():
                    # extract the HF model mean
                    ref_mean = self.pred_data[assim_index][-1][dat].mean(axis=1)
                        # modify each level
                    for level in range(self.tot_level - 1):
                        self.pred_data[assim_index][level][dat] += (ref_mean - self.pred_data[assim_index][level][dat].mean(axis=1))


    def address_ML_error(self):
        print('address_ML_error -- Not yet implemented')
