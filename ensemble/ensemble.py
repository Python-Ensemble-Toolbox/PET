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
import pipt.misc_tools.extract_tools as extract
import pipt.misc_tools.ensemble_tools as entools
import pipt.misc_tools.data_tools as dtools
from misc.system_tools.environ_var import OpenBlasSingleThread  # Single threaded OpenBLAS runs

# Settings
#######################################################################################################
progbar_settings = {
    'ncols': 110,
    'colour': "#285475",
    'bar_format': '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    'ascii': '-◼', # Custom bar characters for a sleeker look
    'unit': 'member',
}
#######################################################################################################

class Ensemble:
    """
    Class for organizing misc. variables and simulator for an ensemble-based inversion run. Here, the forecast step
    and prediction runs are performed. General methods that are useful in various ensemble loops have also been
    implemented here.
    """

    def __init__(self, keys_en: dict, sim, redund_sim=None):
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

        # Initialize some attributes
        self.pred_data = None
        self.enX_temp = None
        self.enX = None
        self.idX = {}

        # Auxilliary input to the simulator - can be used e.g.,
        # to allow for different models when optimizing.
        self.aux_input = None

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
            self.logger('\033[92m--- Restart run initiated! ---\033[92m')
            # Check if the pickle save file exists in folder
            try:
                assert (self.pickle_restart_file in [
                        f for f in os.listdir('.') if os.path.isfile(f)])
            except AssertionError as err:
                self.logger('The restart file "{0}" does not exist in folder. Cannot restart!'.format(
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
            if 'state' in self.keys_en:
                self.prior_info = extract.extract_prior_info(self.keys_en)
            elif 'controls' in self.keys_en:
                self.prior_info = extract.extract_initial_controls(self.keys_en)

            # Calculate initial ensemble if IMPORTSTATICVAR has not been given in init. file.
            # Prior info. on state variables must be given by PRIOR_<STATICVAR-name> keyword.
            if 'importstaticvar' not in self.keys_en:
                self.ne = int(self.keys_en['ne'])

                # Generate prior ensemble
                self.enX, self.idX, self.cov_prior = entools.generate_prior_ensemble(
                    prior_info = self.prior_info, 
                    size = self.ne,
                    save = self.keys_en.get('save_prior', True)
                )

            else:
                # State variable imported as a Numpy save file
                tmp_load = np.load(self.keys_en['importstaticvar'], allow_pickle=True)

                # We assume that the user has saved the state dict. as **state (effectively saved all keys in state
                # individually).
                for key in self.keys_en['staticvar']:
                    if self.enX is None:
                        self.enX = tmp_load[key]
                        self.ne = self.enX.shape[1]
                    else:
                        assert self.ne == tmp_load[key].shape[1], 'Ensemble size of imported state variables do not match!'
                        self.enX = np.vstack((self.enX, tmp_load[key]))

                    # fill in indices
                    self.idX[key] = (self.enX.shape[0] - tmp_load[key].shape[0], self.enX.shape[0])
                
                self.list_states = list(self.keys_en['staticvar'])

        if 'multilevel' in self.keys_en:
            self.multilevel = extract.extract_multilevel_info(self.keys_en['multilevel'])
            self.ml_ne = self.multilevel['ml_ne']
            self.tot_level = len(self.multilevel['levels'])
            self.ml_corr_done = False
        
        
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

    def calc_prediction(self, enX=None, save_prediction=None):
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
        one_state = False

        # Use input state if given
        if enX is None: 
            use_input_ensemble = False
            enX = self.enX
            self.enX = None # free memory
        else:
            use_input_ensemble = True

        if isinstance(enX,list) and hasattr(self, 'multilevel'): # assume multilevel is used if state is a list
            success = self.calc_ml_prediction(enX)
        else:

            # Number of parallel runs
            nparallel = int(self.sim.input_dict.get('parallel', 1))
            self.pred_data = []

            # Run setup function for redund simulator
            if self.sim.redund_sim is not None:
                if hasattr(self.sim.redund_sim, 'setup_fwd_run'):
                    self.sim.redund_sim.setup_fwd_run()
            
            # Run setup function for simulator
            if hasattr(self.sim, 'setup_fwd_run'):
                self.sim.setup_fwd_run(redund_sim=self.sim.redund_sim)

            if enX.ndim == 1:
                one_state = True
                enX = enX[:, np.newaxis]
            elif enX.shape[1] == 1:
                one_state = True

            # If we have several models (num_models) but only one state input
            if one_state and self.ne > 1:
                enX = np.tile(enX, (1, self.ne))
            
            # Convert ensemble matrix to list of dictionaries
            enX = entools.matrix_to_list(enX, self.idX)
                
            if not (self.aux_input is None): 
                for n in range(self.ne):
                    enX[n]['aux_input'] = self.aux_input[n]

            ######################################################################################################################
            # No parralelization
            if nparallel==1:
                en_pred = []
                pbar = tqdm(enumerate(enX), total=self.ne, **progbar_settings)
                for member_index, state in pbar:
                    en_pred.append(self.sim.run_fwd_sim(state, member_index))
            
            # Parallelization on HPC using SLURM
            elif self.sim.input_dict.get('hpc', False): # Run prediction in parallel on hpc
                en_pred = self.run_on_HPC(enX, batch_size=nparallel)

            # Parallelization on local machine using p_map      
            else:
                en_pred = p_map(
                    self.sim.run_fwd_sim, 
                    enX,
                    list(range(self.ne)), 
                    num_cpus=nparallel, 
                    disable=self.disable_tqdm,
                    **progbar_settings
                )
            ######################################################################################################################

            # Convert state enemble back to matrix form
            enX = entools.list_to_matrix(enX, self.idX)

            # If only one state was inputted, keep only that state
            if one_state and self.ne > 1:
                enX = enX[:,0][:,np.newaxis]
            
            # restore state ensemble if it was not inputted
            if not use_input_ensemble:
                self.enX = enX
                enX = None # free memory
            
            # List successful runs and crashes
            success = True
            list_success = [indx for indx, el in enumerate(en_pred) if el is not False]
            list_crash   = [indx for indx, el in enumerate(en_pred) if el is False]
        
            # Dump all information and print error if all runs have crashed
            if not list_success:
                self.save()
                success = False
                if len(list_crash) > 1:
                    print(
                        '\n\033[1;31mERROR: All started simulations has failed! We dump all information and exit!\033[1;m')
                    self.logger(
                        '\n\033[1;31mERROR: All started simulations has failed! We dump all information and exit!\033[1;m')
                    sys.exit(1)
                return success

            # Check crashed runs
            if list_crash:
                # Replace crashed runs with (random) successful runs. If there are more crashed runs than successful once,
                # we draw with replacement.
                if len(list_crash) < len(list_success):
                    copy_member = np.random.choice(list_success, size=len(list_crash), replace=False)
                else:
                    copy_member = np.random.choice(list_success, size=len(list_crash), replace=True)

                # Insert the replaced runs in prediction list
                for index, element in enumerate(copy_member):
                    msg = (
                        f"\033[92m--- Ensemble member {list_crash[index]} failed, "
                        f"has been replaced by ensemble member {element}! ---\033[92m"
                    )
                    print(msg)
                    self.logger(msg)
                    if enX.shape[1] > 1:
                        enX[:, list_crash[index]] = deepcopy(self.enX[:, element])
                    en_pred[list_crash[index]] = deepcopy(en_pred[element])
            
            if hasattr(self.sim, 'compute_adjoints') and self.sim.compute_adjoints:
                en_pred, en_adj = zip(*en_pred)
                self.enGrad = [dtools.melt_adjoint_to_sensitivity(adj, self.sim.datatype) for adj in en_adj]
                #self.enGrad = dtools.combine_adjoint_ensemble(en_adj, self.sim.datatype, self.idX)
                print(self.enGrad)


            self.pred_data = dtools.en_pred_to_pred_data(en_pred)

        # some predicted data might need to be adjusted (e.g. scaled or compressed if it is 4D seis data). Do not
        # include this here.
        if enX is not None:
            self.enX = enX
            enX = None  # free memory

        # Store results if needed
        if save_prediction is not None:
            np.savez(f'{save_prediction}.npz', **{'pred_data': self.pred_data})

        return success
    
    def run_on_HPC(self, enX, batch_size=None, **kwargs):
        list_member_index = list(range(self.ne))

        # Split the ensemble into batches of 500
        if batch_size >= 1000:
            self.logger(f'Cannot run batch size of {batch_size}. Set to 1000')
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
                    zip([enX[curr_n] for curr_n in n_e], [list_member_index[curr_n] for curr_n in n_e])]
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
                        at.store_ensemble_sim_information(self.sim.saveinfo, member_i)
                else:
                    en_pred.append(False)
                self.sim.remove_folder(member_i)
        
        return en_pred

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

    def calc_ml_prediction(self, enX=None):
        """
        Function for running the simulator over several levels. We assume that it is sufficient to provide the level
        integer to the setup of the forward run. This will initiate the correct simulator fidelity.
        The function then runs the set of state through the different simulator fidelities.

        Parameters
        ----------
        enX:
            If simulation is run stand-alone one can input any state.
        """

        no_tot_run = int(self.sim.input_dict['parallel'])
        ml_pred_data = []

        for level in tqdm(self.multilevel['levels'], desc='Fidelity level', position=1, **progbar_settings):

            # Setup forward simulator and redundant simulator at the correct fidelity
            if self.sim.redund_sim is not None:
                if hasattr(self.sim.redund_sim, 'setup_fwd_run'):
                    self.sim.redund_sim.setup_fwd_run(level=level)

            # Run setup function for simulator
            if hasattr(self.sim, 'setup_fwd_run'):
                self.sim.setup_fwd_run(level=level)

            ml_ne = self.multilevel['ne'][level]
            if ml_ne:

                level_enX = entools.matrix_to_list(enX[level], self.idX)
                for n in ml_ne:
                    if self.aux_input is not None:
                        level_enX[n]['aux_input'] = self.aux_input[n]


                # Index list of ensemble members
                list_member_index = list(ml_ne)

                # Run prediction in parallel using p_map
                en_pred = p_map(
                    self.sim.run_fwd_sim, 
                    level_enX,
                    list_member_index, 
                    num_cpus=no_tot_run, 
                    disable=self.disable_tqdm,
                    **progbar_settings,
                )

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
                        self.logger(
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
                    for index, element in enumerate(copy_member):
                        msg = (
                        f"\033[92m--- Ensemble member {list_crash[index]} failed, "
                        f"has been replaced by ensemble member {element}! ---\033[92m"
                        )
                        print(msg)
                        self.logger(msg)
                        if enX[level].shape[1] > 1:
                            enX[level][:, list_crash[index]] = deepcopy(enX[level][:, element])
                        
                        en_pred[list_crash[index]] = deepcopy(en_pred[element])

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
        if self.multilevel['ml_error_corr']:
            scheme = self.multilevel['ml_error_corr'][1]

            if scheme =='sep':
                self.calc_modeling_error_sep()
                self.address_ML_error()
            elif scheme =='once':
                if not self.ml_corr_done:
                    self.calc_modeling_error_ens()
                    self.ml_corr_done = True
                self.address_ML_error()
            elif scheme =='ens':
                self.calc_modeling_error_ens()

    def calc_modeling_error_sep(self):
        print('calc_modeling_error_sep -- Not yet implemented')

    def calc_modeling_error_ens(self):

        if self.multilevel['ml_error_corr'][0] =='bias_corr':
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
