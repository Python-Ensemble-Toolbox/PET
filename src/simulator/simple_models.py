"""A collection of trivial toy models."""
# Imports
import numpy as np  # Misc. numerical tools
import os  # Misc. system tools
import sys
import scipy.stats as sc  # Extended numerical tools
from copy import copy, deepcopy
from multiprocessing import Process, Pipe  # To be able to run Python methods in background
import time  # To wait a bit before loading files

import h5py  # To load matlab .mat files
from scipy import interpolate


class lin_1d:
    """
    linear 1x150 model (or whatever), just make observations of the state at given positions.
    """

    def __init__(self, input_dict=None, m=None):
        """
        Two inputs here. A dictionary of keys, or parameter directly.

        Parameters
        ----------
        input_dict : dict, optional
            Dictionary containing all information required to run the simulator. It may come from, for example, an init file.

        m : int, optional
            Parameter to make predicted data.

        Changelog
        ---------
        - ST 7/9-15
        """
        self.input_dict = input_dict

        assert 'reporttype' in self.input_dict, 'Reporttype is missing, please specify this'
        assert 'reportpoint' in self.input_dict, 'Reportpoint is missing, please specify this'

        self.true_prim = [self.input_dict['reporttype'], self.input_dict['reportpoint']]

        self.true_order = [self.input_dict['reporttype'], self.input_dict['reportpoint']]
        self.all_data_types = self.input_dict['datatype']
        self.l_prim = [int(i) for i in range(len(self.true_prim[1]))]

        # Inputs
        self.input_dict = input_dict
        self.m = m
        self.keys = {}

    def setup_fwd_run(self, **kwargs):
        self.__dict__.update(kwargs)  # parse kwargs input into class attributes
        assimIndex = [i for i in range(len(self.l_prim))]
        trueOrder = self.true_order

        self.pred_data = [deepcopy({}) for _ in range(len(assimIndex))]
        for ind in self.l_prim:
            for key in self.all_data_types:
                self.pred_data[ind][key] = np.zeros((1, 1))

        if isinstance(trueOrder[1], list):  # Check if true data prim. ind. is a list
            self.true_prim = [trueOrder[0], [x for x in trueOrder[1]]]
        else:  # Float
            self.true_prim = [trueOrder[0], [trueOrder[1]]]

    def run_fwd_sim(self, state, member_i, del_folder=True):
        inv_param = state.keys()
        for prim_ind in self.l_prim:
            for dat in self.all_data_types:
                tmp_val = []
                for para in inv_param:
                    tmp_val.append(state[para][self.true_prim[1][prim_ind]])
                self.pred_data[prim_ind][dat] = np.array(tmp_val)

        return self.pred_data


class nonlin_onedimmodel:
    """
    Class of simple 1D forward model for testing purposes.
    """

    def __init__(self, input_dict=None):
        """
        Two inputs here. A dictionary of keys, or parameter directly.

        Parameters
        ----------
        input_dict: dict, optional
            contains all information the run the simulator (may come from, e.g., an init file)
        """
        self.input_dict = input_dict

        assert 'reporttype' in self.input_dict, 'Reporttype is missing, please specify this'
        assert 'reportpoint' in self.input_dict, 'Reportpoint is missing, please specify this'

        self.true_prim = [self.input_dict['reporttype'], self.input_dict['reportpoint']]

        self.true_order = [self.input_dict['reporttype'], self.input_dict['reportpoint']]
        self.all_data_types = self.input_dict['datatype']
        self.l_prim = [int(i) for i in range(len(self.true_prim[1]))]

    def setup_fwd_run(self, **kwargs):
        self.__dict__.update(kwargs)  # parse kwargs input into class attributes
        assimIndex = [i for i in range(len(self.l_prim))]
        trueOrder = self.true_order

        self.pred_data = [deepcopy({}) for _ in range(len(assimIndex))]
        for ind in self.l_prim:
            for key in self.all_data_types:
                self.pred_data[ind][key] = np.zeros((1, 1))

        if isinstance(trueOrder[1], list):  # Check if true data prim. ind. is a list
            self.true_prim = [trueOrder[0], [x for x in trueOrder[1]]]
        else:  # Float
            self.true_prim = [trueOrder[0], [trueOrder[1]]]

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # Fwd. model given by Chen & Oliver, Computat. Geosci., 17(4), p. 689-703, 2013.
        inv_param = state.keys()
        for prim_ind in self.l_prim:
            for dat in self.all_data_types:
                tmp_val = []
                for para in inv_param:
                    tmp_val.append(
                        (7 / 12) * (state[para] ** 3) - (7 / 2) * (state[para] ** 2) + 8 * state[para])
                self.pred_data[prim_ind][dat] = np.array(tmp_val)

        return self.pred_data


class sevenmountains:
    """
    The objective function is the elevations of the seven mountains around bergen, to test optimization algorithm
    """

    def __init__(self, input_dict=None, state=None):
        """
        Two inputs here. A dictionary of keys, or parameter directly.

        Parameters
        ----------
        input_dict : dict, optional
            contains all information the run the simulator (may come from, e.g., an init file)
        state : any, optional
            Parameter to make predicted data

        Changelog
        ---------
        - ST 9/5-18
        """

        # Inputs
        self.input_dict = input_dict

        self._load_sevenmountains_data()

        # If input option 1 is selected
        if self.input_dict is not None:
            self._extInfoInputDict()

        self.state = state

        # Within
        self.m_inv = None

    def _load_sevenmountains_data(self):
        # load mountain coordinates from .mat file
        with h5py.File('sevenMountains.mat', 'r') as f:
            longitude = list(f['X'])
            self.longitude = longitude[0]
            latitude = list(f['Y'])
            self.latitude = latitude[0]
            elevation = list(f['Z'])
            self.elevation = np.asarray(elevation)
        # self.elevation = interpolate.interp2d(longitude, latitude, elevation)

    def _extInfoInputDict(self):
        """
        Extract the manditory and optional information from the input_dict dictionary.


        Parameters
        ----------
        input_dict : dict
            Dictionary containing all information required to run the simulator. (Defined in self)

        Returns
        -------
        filename : str
            Name of the .mako file utilized to generate the ecl input .DATA file.

        Changelog
        ---------
        - KF 14/9-2015
        """
        # In the ecl framework, all reference to the filename should be uppercase
        # self.file = self.input_dict['runfile'].upper()

        # SIMOPTIONS - simulator options
        # TODO: Change this when more options are implemented
        if 'simoption' in self.input_dict and self.input_dict['simoptions'][0] == 'sim_path':
            self.options = {'sim_path': self.input_dict['simoptions'][1]}
        else:
            self.options = {'sim_path': ''}

        if 'origbounds' in self.input_dict:
            if isinstance(self.input_dict['origbounds'][0], list):
                origbounds = np.array(self.input_dict['origbounds'])
            elif self.input_dict['origbounds'] == 'auto':
                origbounds = np.array([[np.min(self.longitude), np.max(self.longitude)],
                                       [np.min(self.latitude), np.max(self.latitude)]])
            else:
                origbounds = np.array([self.input_dict['origbounds']])
            self.orig_lb = origbounds[:, 0]
            self.orig_ub = origbounds[:, 1]

        if 'obj_const' in self.input_dict:
            self.obj_scaling = self.input_dict['obj_const'][0][1]

        # if
        # # load mountain coordinates from .mat file and interpolate
        # with h5py.File('sevenMountains.mat', 'r') as f:
        #     longitude = list(f['X'])
        #     longitude = longitude[0]
        #     latitude = list(f['Y'])
        #     latitude = latitude[0]
        #     elevation = list(f['Z'])
        #     elevation = asarray(elevation)
        # self.elevation = interpolate.interp2d(longitude, latitude, elevation)

    def setup_fwd_run(self, state, assim_ind=None, true_ind=None):
        """
        Set input parameters from an fwd_sim in the simulation to get predicted data. Parameters can be an ensemble or
        a single array.

        Parameters
        ----------
        state : dict
            Dictionary of input parameter. It can be either a single 'state' or an ensemble of 'state'.

        Other Parameters
        ----------------
        true_ind : list
            The list of observed data assimilation indices.

        assim_ind : list
            List with information on assimilation order for ensemble-based methods.

        Changelog
        ---------
        - ST 3/6-16
        """

        # Extract parameter
        self.m_inv = state['coord']

    def run_fwd_sim(self, en_member=None, folder=os.getcwd(), wait_for_proc=False):
        """
        Set up and run a forward simulation in an fwd_sim. The parameters for the forward simulation is set in
        setup_fwd_run. All the steps to set up and run a forward simulation is done in this object.

        Parameters
        ----------
        en_member : int, optional
            Index of the ensemble member to be run.

        folder : str, optional
            Folder where the forward simulation is run.

        Changelog
        ---------
        - ST 3/6-16
        """
        # Set free parameter to fixed
        if self.m_inv.ndim > 1:  # Ensemble matrix
            self.state = self.m_inv[:, en_member].copy()
        else:  # Deterministic fwd_sim
            self.state = copy(self.m_inv)

        # Run forward model
        # Use Process to run the scripts needed in the simulation
        p = Process(target=self.call_sim, args=(folder,))
        p.start()
        if wait_for_proc is True:  # Use the join method of Process to wait for simulation to be finished
            p.join()
        else:
            pass

        return p

    def call_sim(self, path=None):
        """
        Run the simple 1D forward model

        Parameters
        ----------
        path : str, optional
            Alternative folder where the MARE2DEM input files are located.

        Returns
        -------
        d : object
            Predicted data.

        Changelog
        ---------
        - ST 3/6-16
        """
        # Save path
        if path is not None:
            filename = path
        else:
            filename = ''

        func = interpolate.interp2d(self.longitude, self.latitude, self.elevation)

        # Convert to original scale
        # for i, key in enumerate(self.state):
        control = self.orig_lb + self.state * (self.orig_ub - self.orig_lb)

        if control.ndim == 1:
            d = func(control[0], control[1])
        elif control.ndim == 2:
            n = control.shape[1]
            d = []
            for i in range(n):
                d[i] = func(control[0][i], control[1][i])
        else:
            print('\033[1;31mERROR: Input to objective function has wrong dimension.\033[1;m')
            sys.exit(1)
        # # Calc. data
        # d = -self.m ** 2

        # Save in a numpy zip file
        np.savez(filename + 'pred_data.npz', d=[d])

    # Create static method since the following function does not use 'self'
    @staticmethod
    def get_sim_results(which_resp, ext_data_info=None, member=None):
        """
        Get forward simulation results. Simply load the numpy array...

        Parameters
        ----------
        which_resp : str
            Specifies which of the responses is to be outputted (just one data type in this case).

        member : int, optional
            Ensemble member that is finished.

        Returns
        -------
        y : numpy.ndarray
            Array containing the predicted data (response).

        Changelog
        ---------
        - ST 3/6-16
        """

        # Ensemble runs
        if member is not None:
            filename = 'En_' + str(member) + os.sep

        # Single run
        else:
            filename = ''

        # Load file and get result
        time.sleep(0.1)
        load_file = np.load(filename + 'pred_data.npz')
        y = np.squeeze(load_file['d'])

        # Return predicted data
        return y

    # Create static method since the following function does not use 'self'
    @staticmethod
    def get_obj_func(obj_func_name, data_info=None, member=None):
        # Ensemble runs
        if member is not None:
            filename = 'En_' + str(member) + os.sep

        # Single run
        else:
            filename = ''

        # Load file and get result
        time.sleep(0.1)
        load_file = np.load(filename + 'pred_data.npz')
        y = np.squeeze(load_file['d'])

        # Return predicted data
        return y

    # Create static method since the following function does not use 'self'
    @staticmethod
    def check_sim_end(current_run):
        """
        Check if a simulation that has run in the background is finished. For ensemble-based methods,
        there can possibly be several folders with simulations, and each folder must be checked for finished runs. To
        check if simulation is done we search for .resp file which is the output in a successful  run.

        Parameters
        ----------
        current_run : list
            List of ensemble members currently running simulation.

        Returns
        -------
        member : int
            Ensemble member that is finished.

        Changelog
        ---------
        - ST 9/5-18
        """

        # Initialize output
        member = None

        # Search for .resp file
        if isinstance(current_run, list):
            for i in range(len(current_run)):  # Check all En_ folders
                # Search with a specific En_folder
                for file in os.listdir('En_' + str(current_run[i])):
                    if file == 'pred_data.npz':  # If there is a output .npz file
                        member = current_run[i]
        else:
            member = current_run

        return member


class noSimulation:

    def __init__(self, input_dict):
        # parse information from the input.
        # Needs to get datatype, reporttype and reportpoint
        self.input_dict = input_dict
        self.true_order = None

    def setup_fwd_run(self, **kwargs):
        # do whatever initialization you need.
        # Useful to initialize the self.pred_data variable.
        # self.pred_data is a list of dictionaries. Where each list element represents
        # a reportpoint and the dictionary should have the datatypes as keys.
        # Entries in the dictionary are numpy arrays.
        self.__dict__.update(kwargs)  # parse kwargs input into class attributes

    def run_fwd_sim(self, state, member):
        # run simulator. Called from the main function using p_map from p_tqdm package.
        # Return pred_data if run is successfull, False if run failed.
        return [state]
