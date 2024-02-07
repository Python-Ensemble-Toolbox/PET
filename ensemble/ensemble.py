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
from pipt.geostat.decomp import Cholesky  # Making realizations
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
        if 'restart' in self.keys_en and self.keys_en['restart'] == 'yes':
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
            self._ext_prior_info()

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
            for i, opt in enumerate(list(zip(*self.keys_en['multilevel']))[0]):
                if opt == 'levels':
                    self.multilevel['levels'] = [elem for elem in range(
                        int(self.keys_en['multilevel'][i][1]))]
                if opt == 'en_size':
                    self.multilevel['ne'] = [range(int(el))
                                             for el in self.keys_en['multilevel'][i][1]]

    def _ext_prior_info(self):
        """
        Extract prior information on STATE from keyword(s) PRIOR_<STATE entries>.
        """
        # Parse prior info on each state entered in STATE.
        # Store names given in STATE
        if not isinstance(self.keys_en['state'], list):  # Single string
            state_names = [self.keys_en['state']]
        else:  # List
            state_names = self.keys_en['state']

        # Check if PRIOR_<state names> exists for each entry in STATE
        for name in state_names:
            assert 'prior_' + name in self.keys_en, \
                'PRIOR_{0} is missing! This keyword is needed to make initial ensemble for {0} entered in ' \
                'STATE'.format(name.upper())

        # Init. prior info variable
        self.prior_info = {keys: None for keys in state_names}

        # Loop over each prior keyword and make an initial. ensemble for each state in STATE,
        # which is subsequently stored in the state dictionary. If 3D grid dimensions are inputted, information for
        # each layer must be inputted, else the single information will be copied to all layers.
        grid_dim = np.array([0, 0])
        for name in state_names:
            # initiallize an empty dictionary inside the dictionary.
            self.prior_info[name] = {}
            # List the option names inputted in prior keyword
            # opt_list = list(zip(*self.keys_da['prior_'+name]))
            mean = None
            self.prior_info[name]['mean'] = mean
            vario = [None]
            self.prior_info[name]['vario'] = vario
            aniso = [None]
            self.prior_info[name]['aniso'] = aniso
            angle = [None]
            self.prior_info[name]['angle'] = angle
            corr_length = [None]
            self.prior_info[name]['corr_length'] = corr_length
            self.prior_info[name]['nx'] = self.prior_info[name]['ny'] = self.prior_info[name]['nz'] = None

            # Extract info. from the prior keyword
            for i, opt in enumerate(list(zip(*self.keys_en['prior_' + name]))[0]):
                if opt == 'vario':  # Variogram model
                    if not isinstance(self.keys_en['prior_' + name][i][1], list):
                        vario = [self.keys_en['prior_' + name][i][1]]
                    else:
                        vario = self.keys_en['prior_' + name][i][1]
                elif opt == 'mean':  # Mean
                    mean = self.keys_en['prior_' + name][i][1]
                elif opt == 'var':  # Variance
                    if not isinstance(self.keys_en['prior_' + name][i][1], list):
                        variance = [self.keys_en['prior_' + name][i][1]]
                    else:
                        variance = self.keys_en['prior_' + name][i][1]
                elif opt == 'aniso':  # Anisotropy factor
                    if not isinstance(self.keys_en['prior_' + name][i][1], list):
                        aniso = [self.keys_en['prior_' + name][i][1]]
                    else:
                        aniso = self.keys_en['prior_' + name][i][1]
                elif opt == 'angle':  # Anisotropy angle
                    if not isinstance(self.keys_en['prior_' + name][i][1], list):
                        angle = [self.keys_en['prior_' + name][i][1]]
                    else:
                        angle = self.keys_en['prior_' + name][i][1]
                elif opt == 'range':  # Correlation length
                    if not isinstance(self.keys_en['prior_' + name][i][1], list):
                        corr_length = [self.keys_en['prior_' + name][i][1]]
                    else:
                        corr_length = self.keys_en['prior_' + name][i][1]
                elif opt == 'grid':  # Grid dimensions
                    grid_dim = self.keys_en['prior_' + name][i][1]
                elif opt == 'limits':  # Truncation values
                    limits = self.keys_en['prior_' + name][i][1:]
                elif opt == 'active':  # Number of active cells (single number)
                    active = self.keys_en['prior_' + name][i][1]

            # Check if mean needs to be loaded, or if loaded
            if type(mean) is str:
                assert mean.endswith('.npz'), 'File name does not end with \'.npz\'!'
                load_file = np.load(mean)
                assert len(load_file.files) == 1, \
                    'More than one variable located in {0}. Only the mean vector can be stored in the .npz file!' \
                    .format(mean)
                mean = load_file[load_file.files[0]]
            else:  # Single number inputted, make it a list if not already
                if not isinstance(mean, list):
                    mean = [mean]

            # Check if limits exists
            try:
                limits
            except NameError:
                limits = None

            # check if active exists
            try:
                active
            except NameError:
                active = None

            # Extract x- and y-dim
            nx = int(grid_dim[0])
            ny = int(grid_dim[1])

            # Check if 3D grid inputted. If so, we check if info. has been given on all layers. In the case it has
            # not been given, we just copy the info. given.
            if len(grid_dim) == 3 and grid_dim[2] > 1:  # 3D
                nz = int(grid_dim[2])

                # Check mean when values have been inputted directly (not when mean has been loaded)
                if isinstance(mean, list) and len(mean) < nz:
                    # Check if it is more than one entry and give error
                    assert len(mean) == 1, \
                        'Information from MEAN has been given for {0} layers, whereas {1} is needed!' \
                        .format(len(mean), nz)

                    # Only 1 entry; copy this to all layers
                    print(
                        '\033[1;33mSingle entry for MEAN will be copied to all {0} layers\033[1;m'.format(nz))
                    self.prior_info[name]['mean'] = mean * nz

                else:
                    self.prior_info[name]['mean'] = mean

                # Check variogram model
                if len(vario) < nz:
                    # Check if it is more than one entry and give error
                    assert len(vario) == 1, \
                        'Information from VARIO has been given for {0} layers, whereas {1} is needed!' \
                        .format(len(vario), nz)

                    # Only 1 entry; copy this to all layers
                    print(
                        '\033[1;33mSingle entry for VARIO will be copied to all {0} layers\033[1;m'.format(nz))
                    self.prior_info[name]['vario'] = vario * nz

                else:
                    self.prior_info[name]['vario'] = vario

                # Variance
                if len(variance) < nz:
                    # Check if it is more than one entry and give error
                    assert len(variance) == 1, \
                        'Information from VAR has been given for {0} layers, whereas {1} is needed!' \
                        .format(len(variance), nz)

                    # Only 1 entry; copy this to all layers
                    print(
                        '\033[1;33mSingle entry for VAR will be copied to all {0} layers\033[1;m'.format(nz))
                    self.prior_info[name]['variance'] = variance * nz

                else:
                    self.prior_info[name]['variance'] = variance

                # Aniso factor
                if len(aniso) < nz:
                    # Check if it is more than one entry and give error
                    assert len(aniso) == 1, \
                        'Information from ANISO has been given for {0} layers, whereas {1} is needed!' \
                        .format(len(aniso), nz)

                    # Only 1 entry; copy this to all layers
                    print(
                        '\033[1;33mSingle entry for ANISO will be copied to all {0} layers\033[1;m'.format(nz))
                    self.prior_info[name]['aniso'] = aniso * nz

                else:
                    self.prior_info[name]['aniso'] = aniso

                # Aniso factor
                if len(angle) < nz:
                    # Check if it is more than one entry and give error
                    assert len(angle) == 1, \
                        'Information from ANGLE has been given for {0} layers, whereas {1} is needed!' \
                        .format(len(angle), nz)

                    # Only 1 entry; copy this to all layers
                    print(
                        '\033[1;33mSingle entry for ANGLE will be copied to all {0} layers\033[1;m'.format(nz))
                    self.prior_info[name]['angle'] = angle * nz

                else:
                    self.prior_info[name]['angle'] = angle

                # Corr. length
                if len(corr_length) < nz:
                    # Check if it is more than one entry and give error
                    assert len(corr_length) == 1, \
                        'Information from RANGE has been given for {0} layers, whereas {1} is needed!' \
                        .format(len(corr_length), nz)

                    # Only 1 entry; copy this to all layers
                    print(
                        '\033[1;33mSingle entry for RANGE will be copied to all {0} layers\033[1;m'.format(nz))
                    self.prior_info[name]['corr_length'] = corr_length * nz

                else:
                    self.prior_info[name]['corr_length'] = corr_length

                # Limits, if exists
                if limits is not None:
                    if isinstance(limits[0], list) and len(limits) < nz or \
                            not isinstance(limits[0], list) and len(limits) < 2 * nz:
                        # Check if it is more than one entry and give error
                        assert (isinstance(limits[0], list) and len(limits) == 1), \
                            'Information from LIMITS has been given for {0} layers, whereas {1} is needed!' \
                            .format(len(limits), nz)
                        assert (not isinstance(limits[0], list) and len(limits) == 2), \
                            'Information from LIMITS has been given for {0} layers, whereas {1} is needed!' \
                            .format(len(limits) / 2, nz)

                        # Only 1 entry; copy this to all layers
                        print(
                            '\033[1;33mSingle entry for RANGE will be copied to all {0} layers\033[1;m'.format(nz))
                        self.prior_info[name]['limits'] = [limits] * nz

            else:  # 2D grid only, or optimization case
                nz = 1
                self.prior_info[name]['mean'] = mean
                self.prior_info[name]['vario'] = vario
                self.prior_info[name]['variance'] = variance
                self.prior_info[name]['aniso'] = aniso
                self.prior_info[name]['angle'] = angle
                self.prior_info[name]['corr_length'] = corr_length
                if limits is not None:
                    self.prior_info[name]['limits'] = limits
                if active is not None:
                    self.prior_info[name]['active'] = active

            self.prior_info[name]['nx'] = nx
            self.prior_info[name]['ny'] = ny
            self.prior_info[name]['nz'] = nz

        # Loop over keys and input

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
            nz = self.prior_info[name]['nz']
            mean = self.prior_info[name]['mean']
            nx = self.prior_info[name]['nx']
            ny = self.prior_info[name]['ny']
            if nx == ny == 0:  # assume ensemble will be generated elsewhere if dimensions are zero
                break
            variance = self.prior_info[name]['variance']
            corr_length = self.prior_info[name]['corr_length']
            aniso = self.prior_info[name]['aniso']
            vario = self.prior_info[name]['vario']
            angle = self.prior_info[name]['angle']
            if 'limits' in self.prior_info[name]:
                limits = self.prior_info[name]['limits']
            else:
                limits = None

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
        input_state:
            Use an input state instead of internal state (stored in self) to run predictions
        save_prediction:
            Save the predictions as a <save_prediction>.npz file (numpy compressed file)

        Returns
        -------
        prediction:
            List of dictionaries with keys equal to data types (in DATATYPE),
            containing the responses at each time step given in PREDICTION.

        """

        if hasattr(self, 'multilevel'):
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

        self.treat_modeling_error(self.iteration)

        return success
