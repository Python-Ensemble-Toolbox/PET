"""Wrap Eclipse"""
# External imports
import numpy as np
import sys
import os
from copy import deepcopy
from mako.lookup import TemplateLookup
from mako.runtime import Context
from multiprocessing import Process
import datetime as dt
from scipy import interpolate
from subprocess import call, DEVNULL
from misc import ecl, grdecl
from shutil import rmtree, copytree  # rmtree for removing folders
import time
# import rips

# Internal imports
from misc.system_tools.environ_var import EclipseRunEnvironment
from pipt.misc_tools.analysis_tools import store_ensemble_sim_information


class eclipse:
    """
    Class for running the Schlumberger eclipse 100 black oil reservoir simulator. For more information see  GeoQuest:
    ECLIPSE reference manual 2009.1. Schlumberger, GeoQuest (2009).

    To run this class, eclipse must be installed and elcrun must be in the system path!
    """

    def __init__(self, input_dict=None, filename=None, options=None):
        """
        The inputs are all optional, but in the same fashion as the other simulators a system must be followed.
        The input_dict can be utilized as a single input. Here all nescessary info is stored. Alternatively,
        if input_dict is not defined, all the other input variables must be defined.

        Parameters
        ----------
        input_dict : dict, optional
            Dictionary containing all information required to run the simulator.

                - parallel: number of forward simulations run in parallel
                - simoptions: options for the simulations
                    - mpi: option to use mpi (always use > 2 cores)
                    - sim_path: Path to the simulator
                    - sim_flag: Flags sent to the simulator (see simulator documentation for all possibilities)
                - sim_limit: maximum number of seconds a simulation can run before being killed
                - runfile: name of the simulation input file
                - reportpoint: these are the dates the simulator reports results
                - reporttype: this key states that the report poins are given as dates
                - datatype: the data types the simulator reports
                - replace: replace failed simulations with randomly selected successful ones
                - rerun: in case of failure, try to rerun the simulator the given number of times
                - startdate: simulaton start date
                - saveforecast: save the predicted measurements for each iteration

        filename : str, optional
            Name of the .mako file utilized to generate the ECL input .DATA file. Must be in uppercase for the
            ECL simulator.

        options : dict, optional
            Dictionary with options for the simulator.

        Returns
        -------
        initial_object : object
            Initial object from the class ecl_100.
        """

        # IN
        self.input_dict = input_dict
        self.file = filename
        self.options = options

        self.upscale = None
        # If input option 1 is selected
        if self.input_dict is not None:
            self._extInfoInputDict()

        # Allocate for internal use
        self.inv_stat = None
        self.static_state = None

    def _extInfoInputDict(self):
        """
        Extract the manditory and optional information from the input_dict dictionary.

        Parameters
        ----------
        input_dict : dict
            Dictionary containing all information required to run the simulator (defined in self).

        Returns
        -------
        filename : str
            Name of the .mako file utilized to generate the ECL input .DATA file.
        """
        # Chech for mandatory keys
        assert 'reporttype' in self.input_dict, 'Reporttype is missing, please specify this'
        assert 'reportpoint' in self.input_dict, 'Reportpoint is missing, please specify this'

        self.true_prim = [self.input_dict['reporttype'], self.input_dict['reportpoint']]

        self.true_order = [self.input_dict['reporttype'], self.input_dict['reportpoint']]
        self.all_data_types = self.input_dict['datatype']
        self.l_prim = [int(i) for i in range(len(self.true_prim[1]))]

        # In the ecl framework, all reference to the filename should be uppercase
        self.file = self.input_dict['runfile'].upper()
        self.options = {}
        self.options['sim_path'] = ''
        self.options['sim_flag'] = ''
        self.options['mpi'] = ''
        self.options['parsing-strictness'] = ''
        # Loop over options in SIMOPTIONS and extract the parameters we want
        if 'simoptions' in self.input_dict:
            if type(self.input_dict['simoptions'][0]) == str:
                self.input_dict['simoptions'] = [self.input_dict['simoptions']]
            for i, opt in enumerate(list(zip(*self.input_dict['simoptions']))[0]):
                if opt == 'sim_path':
                    self.options['sim_path'] = self.input_dict['simoptions'][i][1]
                if opt == 'sim_flag':
                    self.options['sim_flag'] = self.input_dict['simoptions'][i][1]
                if opt == 'mpi':
                    self.options['mpi'] = self.input_dict['simoptions'][i][1]
                if opt == 'parsing-strictness':
                    self.options['parsing-strictness'] = self.input_dict['simoptions'][i][1]
        if 'sim_limit' in self.input_dict:
            self.options['sim_limit'] = self.input_dict['sim_limit']

        if 'reportdates' in self.input_dict:
            self.reportdates = [
                x * 30 for x in range(1, int(self.input_dict['reportdates'][1]))]

        if 'read_sch' in self.input_dict:
            # self.read_sch = self.input_dict['read_sch'][0]
            load_file = np.load(self.input_dict['read_sch'], allow_pickle=True)
            self.reportdates = load_file[load_file.files[0]]

        if 'startdate' in self.input_dict:
            self.startDate = {}
            # assume date is on form day/month/year
            tmpDate = [int(elem) for elem in self.input_dict['startdate'].split('/')]

            self.startDate['day'] = tmpDate[0]
            self.startDate['month'] = tmpDate[1]
            self.startDate['year'] = tmpDate[2]

        if 'realizations' in self.input_dict:
            self.realizations = self.input_dict['realizations']

        if 'trunc_level' in self.input_dict:
            self.trunc_level = self.input_dict['trunc_level']

        if 'rerun' in self.input_dict:
            self.rerun = int(self.input_dict['rerun'])
        else:
            self.rerun = 0

        # If we want to extract, or evaluate, something uniquely from the ensemble specific run we can
        # run a user defined code to do this.
        self.saveinfo = None
        if 'savesiminfo' in self.input_dict:
            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(self.input_dict['savesiminfo'], list):
                self.saveinfo = self.input_dict['savesiminfo']
            else:
                self.saveinfo = [self.input_dict['savesiminfo']]

        if 'upscale' in self.input_dict:
            self.upscale = {}
            for i in range(0, len(self.input_dict['upscale'])):
                if self.input_dict['upscale'][i][0] == 'state':
                    # Set the parameter we upscale with regards to, must be the same as one free parameter
                    self.upscale['state'] = self.input_dict['upscale'][i][1]
                    # Set the dimension of the parameterfield to be upscaled (x and y)
                    self.upscale['dim'] = self.input_dict['upscale'][i][2]
                if self.input_dict['upscale'][i][0] == 'maxtrunc':
                    # Set the ratio for the maximum truncation value
                    self.upscale['maxtrunc'] = self.input_dict['upscale'][i][1]
                if self.input_dict['upscale'][i][0] == 'maxdiff':
                    # Set the ratio for the maximum truncation of differences
                    self.upscale['maxdiff'] = self.input_dict['upscale'][i][1]
                if self.input_dict['upscale'][i][0] == 'wells':
                    # Set the list of well indexes, this is a list of lists where each element in the outer list gives
                    # a well coordinate (x and y) as elements in the inner list.
                    self.upscale['wells'] = []
                    for j in range(1, len(self.input_dict['upscale'][i])):
                        self.upscale['wells'].append(
                            [int(elem) for elem in self.input_dict['upscale'][i][j]])
                if self.input_dict['upscale'][i][0] == 'radius':
                    # List of radius lengths
                    self.upscale['radius'] = [int(elem)
                                              for elem in self.input_dict['upscale'][i][1]]
                if self.input_dict['upscale'][i][0] == 'us_type':
                    self.upscale['us_type'] = self.input_dict['upscale'][i][1]

            # Check that we have as many radius elements as wells
            if 'radius' in self.upscale:
                if len(self.upscale['radius']) != len(self.upscale['wells']):
                    sys.exit('ERROR: Missmatch between number of well inputs and number of radius elements. Please check '
                             'the input file')
            else:
                self.upscale['radius'] = []
                self.upscale['wells'] = []

        # The simulator should run on different levels
        if 'multilevel' in self.input_dict:
            # extract list of levels
            self.multilevel = self.input_dict['multilevel']
        else:
            # if not initiallize as list with one element
            self.multilevel = [False]

    def setup_fwd_run(self, **kwargs):
        """
        Setup the simulator.

        Attributes
        ----------
        assimIndex : int
            Gives the index-type (e.g. step,time,etc.) and the index for the
            data to be assimilated
        trueOrder : 
            Gives the index-type (e.g. step,time,etc.) and the index of the true data
        """
        self.__dict__.update(kwargs)  # parse kwargs input into class attributes

        if hasattr(self, 'reportdates'):
            self.report = {'dates': self.reportdates}
        elif 'reportmonths' in self.input_dict:  # for optimization
            self.report = {
                'days': [30 * i for i in range(1, int(self.input_dict['reportmonths'][1]))]}
        else:
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
        # self.all_data_types = list(pred_data[0].keys())

        # Initiallise space to store the number of active cells. This is only for the upscaling option.
        if 'upscale' in self.input_dict:
            self.num_act = []

        # Initialise error summary
        self.error_smr = []

        # Initiallize run time summary
        self.run_time = []

        # Check that the .mako file is in the current working directory
        if not os.path.isfile('%s.mako' % self.file):
            sys.exit(
                'ERROR: .mako file is not in the current working directory. This file must be defined')

    def run_fwd_sim(self, state, member_i, del_folder=True):
        """
        Setup and run the ecl_100 forward simulator. All the parameters are defined as attributes, and the name of the
        parameters are initialized in setupFwdRun. This method sets up and runs all the individual ensemble members.
        This method is based on writing .DATA file for each ensemble member using the mako template, for more info
        regarding mako see http://www.makotemplates.org/

        Parameters
        ----------
        state : dict
            Dictionary containing the ensemble state.

        member_i : int
            Index of the ensemble member.

        del_folder : bool, optional
            Boolean to determine if the ensemble folder should be deleted. Default is False.
        """
        if hasattr(self, 'level'):
            state['level'] = self.level
        else:
            state['level'] = 0  # default value
        os.mkdir('En_' + str(member_i))
        folder = 'En_' + str(member_i) + os.sep

        state['member'] = member_i
        # If the run is upscaled, run the upscaling procedure
        if self.upscale is not None:
            if hasattr(self, 'level'):  # if this is a multilevel run, we must set the level
                self.upscale['maxtrunc'] = self.trunc_level[self.level]
            if self.upscale['maxtrunc'] > 0:  # if the truncation level is 0, we do not perform upscaling
                self.coarsen(folder, state)
            # if the level is 0, and upscaling has been performed earlier this must be
            elif hasattr(self, 'coarse'):
                # removed
                del self.coarse

        # start by generating the .DATA file, using the .mako template situated in ../folder
        self._runMako(folder, state)
        success = False
        rerun = self.rerun
        while rerun >= 0 and not success:
            success = self.call_sim(folder, True)
            rerun -= 1
        if success:
            self.extract_data(member_i)
            if del_folder:
                if self.saveinfo is not None:  # Try to save information
                    store_ensemble_sim_information(self.saveinfo, member_i)
                self.remove_folder(member_i)
            return self.pred_data
        else:
            if self.redund_sim is not None:
                success = self.redund_sim.call_sim(folder, True)
                if success:
                    self.extract_data(member_i)
                    if del_folder:
                        if self.saveinfo is not None:  # Try to save information
                            store_ensemble_sim_information(self.saveinfo, member_i)
                        self.remove_folder(member_i)
                    return self.pred_data
                else:
                    if del_folder:
                        self.remove_folder(member_i)
                    return False
            else:
                if del_folder:
                    self.remove_folder(member_i)
                return False

    def remove_folder(self, member):
        folder = 'En_' + str(member) + os.sep
        try:
            rmtree(folder)  # Try to delete folder
        except:  # If deleting fails, just rename to 'En_<ensembleMember>_date'
            os.rename(folder, folder + '_' +
                      dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    def extract_data(self, member):
        # get the formated data
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if self.pred_data[prim_ind][key] is not None:  # Obs. data at assim. step
                    true_data_info = [self.true_prim[0], self.true_prim[1][prim_ind]]
                    try:
                        data_array = self.get_sim_results(key, true_data_info, member)
                        self.pred_data[prim_ind][key] = data_array
                    except:
                        pass

    def coarsen(self, folder, ensembleMember=None):
        """
        This method utilized one field parameter to upscale the computational grid. A coarsening file is written to the
        ensemble folder, and the eclipse calculates the upscaled permeabilities, porosities and transmissibilities
        based on the new grid and the original parameter values.

        Parameters
        ----------
        folder : str, optional
            Path to the ecl_100 run folder.

        ensembleMember : int, optional
            Index of the ensemble member to run.

        Changelog
        ---------
        - KF 17/9-2015 Added uniform upscaling as an option
        - KF 6/01-17
        """

        # Get the parameter values for the current ensemble member. Note that we select only one parameter
        if ensembleMember is not None:
            coarsenParam = self.inv_state[self.upscale['state']][:, ensembleMember]
        else:
            coarsenParam = self.inv_state[self.upscale['state']]

        if self.upscale['us_type'].lower() == 'haar':  # the upscaling is based on a Haar wavelet
            # Do not add any dead-cells
            # TODO: Make a better method for determining dead or inactive cells.

            orig_wght = np.ones((self.upscale['dim'][0], self.upscale['dim'][1]))
            # Get the new cell structure by a 2-D unbalanced Haar transform.
            wght = self._Haar(coarsenParam.reshape(
                (self.upscale['dim'][0], self.upscale['dim'][1])), orig_wght)

        elif self.upscale['us_type'].lower() == 'unif':
            # This option generates a grid where the cells are upscaled uniformly in a dyadic manner. We utilize the
            # same framework to define the parameters to be coarsened as in the Haar case. Hence, we only need to
            # calculate the wght variable. The level of upscaling can be determined by a fraction, however, for this
            # case the fraction gives the number of upscaling levels, 1 is all possible levels, 0 is no upscaling.
            wght = self._unif((self.upscale['dim'][0], self.upscale['dim'][1]))

        self.write_coarse(folder, wght, coarsenParam.reshape(
            (int(self.upscale['dim'][0]), int(self.upscale['dim'][1]))))

    def write_coarse(self, folder, whgt, image):
        """
        This function writes the include file coarsen to the ecl run. This file tels ECL to coarsen the grid.

        Parameters
        ----------
        folder : str
            Path to the ecl_100 run.

        whgt : float
            Weight of the transformed cells.

        image : array-like
            Original image.

        Changelog
        ---------
        - KF 17/9-2015
        """

        well = self.upscale['wells']
        radius = self.upscale['radius']

        well_cells = self._nodeIndex(image.shape[1], image.shape[0], well, radius)
        coarse = np.array([[True] * image.shape[1]] * image.shape[0])
        tmp = coarse.flatten()
        tmp[well_cells] = False
        coarse = tmp.reshape(image.shape)

        # f = open(folder + 'coarsen.dat', 'w+')
        # f.write('COARSEN \n')
        ecl_coarse = []

        for level in range(len(whgt) - 1, -1, -1):
            weight_at_level = whgt[level]
            x_dim = weight_at_level.shape[0]
            y_dim = weight_at_level.shape[1]

            merged_at_level = weight_at_level[int(x_dim / 2):, :int(y_dim / 2)]

            level_dim = 2 ** (level + 1)

            for i in range(0, merged_at_level.shape[0]):
                for j in range(0, merged_at_level.shape[1]):
                    if merged_at_level[i, j] == 1:
                        # Do this in two stages to avoid errors with the dimensions
                        #
                        # only merging square cells
                        if (i + 1) * level_dim <= image.shape[0] and (j + 1) * level_dim <= image.shape[1]:
                            if coarse[i * level_dim:(i + 1) * level_dim, j * level_dim:(j + 1) * level_dim].all():
                                coarse[i * level_dim:(i + 1) * level_dim,
                                       j * level_dim:(j + 1) * level_dim] = False
                                ecl_coarse.append([j * level_dim + 1, (j + 1) * level_dim,
                                                   i * level_dim + 1, (i + 1) * level_dim, 1, 1, 1, 1, 1])
                                # f.write('%i %i %i %i 1 1 1 1 1 / \n' % (j * level_dim + 1, (j + 1) * level_dim,
                                #                                         i * level_dim + 1, (i + 1) * level_dim))
                        # cells at first edge, non-square
                        if (i + 1)*level_dim > image.shape[0] and (j + 1) * level_dim <= image.shape[1]:
                            if coarse[i * level_dim::, j * level_dim:(j + 1) * level_dim].all():
                                coarse[i * level_dim::, j *
                                       level_dim:(j + 1) * level_dim] = False
                                ecl_coarse.append([j * level_dim + 1, (j + 1) * level_dim,
                                                   i * level_dim + 1, image.shape[0], 1, 1, 1, 1, 1])
                                # f.write('%i %i %i %i 1 1 1 1 1 / \n' % (j * level_dim + 1, (j + 1) * level_dim,
                                #                                         i * level_dim + 1, image.shape[0]))
                        # cells at second edge, non-square
                        if (j + 1)*level_dim > image.shape[1] and (i + 1) * level_dim <= image.shape[0]:
                            if coarse[i * level_dim:(i + 1) * level_dim, j * level_dim::].all():
                                coarse[i * level_dim:(i + 1) * level_dim,
                                       j * level_dim::] = False
                                ecl_coarse.append([j * level_dim + 1, image.shape[1],
                                                   i * level_dim + 1, (i + 1) * level_dim, 1, 1, 1, 1, 1])
                                # f.write('%i %i %i %i 1 1 1 1 1 / \n' % (j * level_dim + 1, image.shape[1],
                                #                                         i * level_dim + 1, (i + 1) * level_dim))
                        # cells at intersection between first and second edge, non-square
                        if (i + 1) * level_dim > image.shape[0] and (j + 1) * level_dim > image.shape[1]:
                            if coarse[i * level_dim::, j * level_dim::].all():
                                coarse[i * level_dim::, j * level_dim::] = False
                                ecl_coarse.append([j * level_dim + 1, image.shape[1],
                                                   i * level_dim + 1, image.shape[0], 1, 1, 1, 1, 1])
                                # f.write('%i %i %i %i 1 1 1 1 1 / \n' % (j * level_dim + 1, image.shape[1],
                                #                                         i * level_dim + 1, image.shape[0]))
        # f.write('/')
        # f.close()
        self.coarse = ecl_coarse

    def _nodeIndex(self, x_dir, y_dir, cells, listRadius):
        # Find the node index for the cells, and the cells in a radius around.
        index = []
        for k in range(0, len(cells)):
            cell_x = cells[k][0] - 1  # Remove one to make python equivalent to ecl format
            cell_y = cells[k][1] - 1  # Remove one to make python equivalent to ecl format
            radius = listRadius[k]
            # tmp_index = [cell_x + cell_y*x_dir]
            tmp_index = []
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    x = cell_x + i
                    y = cell_y + j
                    if (0 <= x < x_dir and 0 <= y < y_dir) and (np.sqrt(i ** 2 + j ** 2) <= radius) \
                            and (x + y * x_dir < y_dir * x_dir):
                        tmp_index.append(x + y * x_dir)
            index.extend(tmp_index)

        return index

    def _Haar(self, image, weights):
        """
        The 2D unconditional Haar transform.

        Parameters
        ----------
        image : array-like
            The original image where the Haar-transform is performed. This could be the permeability or the porosity field.

        orig_weights : array-like
            The original weights of the image. If some cells are inactive or dead, their weight can be set to zero.

        Returns
        -------
        tot_weights : array-like
            A vector/matrix of the same size as the image, with the weights of the new cells. This is utilized for writing
            the coarsening file but is not sufficient to recreate the image.
        """
        # Todo: make an option which stores the transformed field.

        # Define the two truncation levels as a fraction of the larges value in the original image. The fraction is defined
        # through the inputs.
        max_value = self.upscale['maxtrunc'] * np.max(image)
        max_diff = self.upscale['maxdiff'] * (np.max(image) - np.min(image))

        tot_transf = []
        tot_weight = []
        tot_alpha = []
        count = 0
        allow_merge = np.ones(image.shape)
        while image.shape[0] > 1:
            level_alpha = []
            if isinstance(image.shape, tuple):
                # the image has two axes.
                # Perform the transformation for all the coulmns, and then for the transpose of the results

                for axis in range(0, 2):
                    # Initialise the matrix for storing the details and smoothing
                    level_diff_columns = np.empty(
                        (int(np.ceil(len(image[:, 0]) / 2)), len(image[0, :])))
                    # level_diff_rows = np.empty((len(image[:, 0]), int(np.ceil(len(image[0, :])/2))))
                    level_smooth_columns = np.empty(
                        (int(np.ceil(len(image[:, 0]) / 2)), len(image[0, :])))
                    level_weight_columns = np.empty(
                        (int(np.ceil(len(image[:, 0]) / 2)), len(image[0, :])))
                    level_alpha_columns = np.empty(
                        (int(np.ceil(len(image[:, 0]) / 2)), len(image[0, :])))

                    for i in range(0, image.shape[1]):
                        if len(image[:, i]) % 2 == 0:
                            diff = image[1::2, i] - image[::2, i]
                            new_weights = weights[::2, i] + weights[1::2, i]
                            alpha = (weights[1::2, i] / new_weights)
                            smooth = image[::2, i] + alpha * diff

                        else:
                            tmp_img = image[:-1, i]
                            diff = tmp_img[1::2] - tmp_img[::2]

                            tmp_weight = weights[:-1, i]
                            new_weights = tmp_weight[::2] + tmp_weight[1::2]
                            new_weights = np.append(new_weights, weights[-1, i])

                            tmp_img = image[:-1, i]
                            alpha = (weights[1::2, i] / new_weights[:-1])
                            smooth = tmp_img[::2] + alpha * diff
                            smooth = np.append(smooth, image[-1, i])
                            alpha = np.append(alpha, 0)
                            diff = np.append(diff, 0)

                        level_diff_columns[:, i] = diff
                        level_weight_columns[:, i] = new_weights
                        level_smooth_columns[:, i] = smooth
                        level_alpha_columns[:, i] = alpha

                    # image = level_smooth_columns.T
                    image = np.vstack((level_smooth_columns, level_diff_columns)).T
                    weights = np.vstack((level_weight_columns, level_weight_columns)).T
                    level_alpha.append(level_alpha_columns)

                image, weights, allow_merge, end_ctrl = self._haarTrunc(image, weights, max_value, max_diff,
                                                                        allow_merge)

                tot_transf.append(image)
                tot_weight.append(weights)
                tot_alpha.append(level_alpha)

                if end_ctrl:
                    break
                image = image[:image.shape[0] / 2, :image.shape[1] / 2]
                weights = weights[:weights.shape[0] / 2, :weights.shape[1] / 2]

            else:
                if len(image) % 2 == 0:
                    diff = image[1::2] - image[::2]
                    new_weights = weights[::2] + weights[1::2]
                    smooth = image[::2] + (weights[1::2] / new_weights) * diff

                else:
                    tmp_img = image[:-1]
                    diff = tmp_img[1::2] - tmp_img[::2]

                    tmp_weight = weights[:-1]
                    new_weights = tmp_weight[::2] + tmp_weight[1::2]
                    new_weights = np.append(new_weights, weights[-1])

                    tmp_img = image[:-1]
                    smooth = tmp_img[::2] + (weights[1::2] / new_weights[:-1]) * diff
                    smooth = np.append(smooth, image[-1])

                tot_transf.append(smooth)
                tot_weight.append(new_weights)

            # weights = new_weights
            # image = smooth
            count += 1

        return tot_weight

    def _unif(self, dim):
        # calculate the uniform upscaling steps

        min_dim = min(dim)  # find the lowest dimension
        max_levels = 0
        while min_dim/2 >= 1:
            min_dim = min_dim/2
            max_levels += 1

        # conservative choice
        trunc_level = int(np.floor(max_levels*self.upscale['maxtrunc']))

        # all the initial cells are upscaled
        wght = [np.ones(tuple([int(elem) for elem in dim]))]

        for i in range(trunc_level):
            new_dim = [int(np.ceil(elem/2))
                       for elem in wght[i].shape]  # always take a larger dimension
            wght.append(np.ones(tuple(new_dim)))

        return wght

    def _haarTrunc(self, image, weights, max_val, max_diff, merge):
        """
        Function for truncating the wavelets. Based on the max_val and max_diff values this function set the detail
        coaffiecient to zero if the value of this coefficient is below max_diff, and the value of the smooth coefficient
        is below max_val.

        Parameters
        ----------
        image : array-like
            The transformed image.

        weights : array-like
            The weights of the transformed image.

        max_val : float
            Smooth values above this value are not allowed.

        max_diff : float
            Detail coefficients higher than this value are not truncated.

        merge : array-like
            Matrix/vector of booleans defining whether a merge is allowed.

        Returns
        -------
        image : array-like
            Transformed image with truncated coefficients.

        weights : array-like
            Updated weights.

        allow_merge : array-like
            Booleans keeping track of allowed merges.

        end_ctrl : bool
            Boolean to control whether further upscaling is possible.
        """
        # the image will always contain even elements in x and y dir
        x_dir = int(image.shape[0] / 2)
        y_dir = int(image.shape[1] / 2)

        smooth = image[:x_dir, :y_dir]
        smooth_diff = image[x_dir:, :y_dir]
        diff_smooth = image[:x_dir, y_dir:]
        diff_diff = image[x_dir:, y_dir:]

        weights_diff = weights[x_dir:, :y_dir]

        allow_merge = np.array(
            [[False] * y_dir] * x_dir)

        # ensure that last row does not be left behind during non-dyadic upscaling
        if image.shape[0] > merge.shape[0]:
            merge = np.insert(merge, -1, merge[-1, :], axis=0)
        if image.shape[1] > merge.shape[1]:
            merge = np.insert(merge, -1, merge[:, -1], axis=1)

        cand1 = zip(*np.where(smooth < max_val))

        end_ctrl = True

        for elem in cand1:
            # If the method wants to merge cells outside the box the if sentence gives an error, make an exception for
            # this hence do not merge these cells.
            try:
                if abs(smooth_diff[elem]) <= max_diff and abs(diff_smooth[elem]) <= max_diff and \
                        merge[elem[0] * 2, elem[1] * 2] and merge[elem[0] * 2 + 1, elem[1] * 2] and \
                        merge[elem[0] * 2 + 1, elem[1] * 2 + 1] and merge[elem[0] * 2, elem[1] * 2 + 1]:
                    diff_smooth[elem] = 0
                    smooth_diff[elem] = 0
                    diff_diff[elem] = 0
                    weights_diff[elem] = True
                    allow_merge[elem] = True
                    end_ctrl = False
            except:
                pass
        # for elem in cand2:
        #     if smooth[elem] <= max_val:
        #         diff_smooth[elem] = 0
        #         smooth_diff[elem] = 0
        #         diff_diff[elem] = 0

        image[x_dir:, :y_dir] = smooth_diff
        image[:x_dir, y_dir:] = diff_smooth
        image[x_dir:, y_dir:] = diff_diff

        weights[x_dir:, :y_dir] = weights_diff

        return image, weights, allow_merge, end_ctrl

    def _runMako(self, folder, state):
        """
        Read the mako template (.mako) file from ../folder, and render the correct data file (.DATA) in folder.

        Parameters
        ----------
        Folder : str, optional
            Folder for the ecl_100 run.

        ensembleMember : int, optional
            Index of the ensemble member to run.

        Changelog
        ---------
        - KF 14/9-2015
        """

        # Check and add report time
        if hasattr(self, 'report'):
            for key in self.report:
                state[key] = self.report[key]

        # Convert drilling order (float) to drilling queue (integer) - drilling order optimization
        # if "drillingorder" in en_dict:
        #     dorder = en_dict['drillingorder']
        #     en_dict['drillingqueue'] = np.argsort(dorder)[::-1][:len(dorder)]

        # Add startdate
        if hasattr(self, 'startDate'):
            state['startdate'] = dt.datetime(
                self.startDate['year'], self.startDate['month'], self.startDate['day'])

        # Add the coarsing values
        if hasattr(self, 'coarse'):
            state['coarse'] = self.coarse

        # Look for the mako file
        lkup = TemplateLookup(directories=os.getcwd(),
                              input_encoding='utf-8')

        # Get template
        # If we need the E300 run, define a E300 data file with _E300 added to the end.
        if hasattr(self, 'E300'):
            if self.E300:
                tmpl = lkup.get_template('%s.mako' % (self.file + '_E300'))
            else:
                tmpl = lkup.get_template('%s.mako' % self.file)
        else:
            tmpl = lkup.get_template('%s.mako' % self.file)

        # use a context and render onto a file
        with open('{0}{1}'.format(folder + self.file, '.DATA'), 'w') as f:
            ctx = Context(f, **state)
            tmpl.render_context(ctx)

    def get_sim_results(self, whichResponse, ext_data_info=None, member=None):
        """
        Read the output from simulator and store as an array. Optionally, if the DA method is based on an ensemble
        method, the output must be read inside a folder.

        Parameters
        ----------
        whichResponse : str
            Which of the responses is to be outputted (e.g., WBHP PRO-1, WOPR, PRESS, OILSAT, etc).

        ext_data_info : tuple, optional
            Tuple containing the assimilation step information, including the place of assimilation (e.g., which TIME) and the
            index of this assimilation place.

        member : int, optional
            Ensemble member that is finished.

        Returns
        -------
        yFlow : array-like
            Array containing the response from ECL 100. The response type is chosen by the user in options['data_type'].

        Notes
        -----
        - Modified the ecl package to allow reading the summary data directly, hence, we get cell, summary, and field data
        from the ecl package.
        - KF 29/10-2015
        - Modified the ecl package to read RFT files. To obtain, e.g,. RFT pressures form well 'PRO-1" whichResponse
        must be rft_PRESSURE PRO-1
        """
        # Check that we have no trailing spaces
        whichResponse = whichResponse.strip()

        # if ensemble DA method
        if member is not None:
            # Get results
            if hasattr(self, 'ecl_case'):
                # En_XX/YYYY.DATA is the folder setup
                rt_mem = int(self.ecl_case.root.split('/')[0].split('_')[1])
                if rt_mem != member:  # wrong case
                    self.ecl_case = ecl.EclipseCase('En_' + str(member) + os.sep + self.file + '.DATA')
            else:
                self.ecl_case = ecl.EclipseCase('En_' + str(member) + os.sep + self.file + '.DATA')
            if ext_data_info[0] == 'days':
                time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                    dt.timedelta(days=ext_data_info[1])
                dates = self.ecl_case.by_date
                if time not in dates and 'date_slack' in self.input_dict:
                    slack = int(self.input_dict['date_slack'])
                    if slack > 0:
                        v = [el for el in dates if np.abs(
                            (el-time).total_seconds()) < slack]
                        if len(v) > 0:
                            time = v[0]
            else:
                time = ext_data_info[1]

            # Check if the data is a field or well data, by checking if the well is defined
            if len(whichResponse.split(' ')) == 2:
                # if rft, search for rft_
                if 'rft_' in whichResponse:
                    # to speed up when performing the prediction step
                    if hasattr(self, 'rft_case'):
                        rt_mem = int(self.rft_case.root.split('/')[0].split('_')[1])
                        if rt_mem != member:
                            self.rft_case = ecl.EclipseRFT('En_' + str(member) + os.sep + self.file)
                    else:
                        self.rft_case = ecl.EclipseRFT(
                            'En_' + str(member) + os.sep + self.file)
                    # Get the data. Due to formating we can slice the property.
                    rft_prop = self.rft_case.rft_data(well=whichResponse.split(
                        ' ')[1], prop=whichResponse.split(' ')[0][4:])
                    # rft_data are collected for open connections. This may vary throughout the simulation, hence we
                    # must also collect the depth for the rft_data to check if all data is present
                    rft_depth = self.rft_case.rft_data(
                        well=whichResponse.split(' ')[1], prop='DEPTH')
                    # to check this we import the referance depth if this is available. If not we assume that the data
                    # is ok.
                    try:
                        ref_depth_f = np.load(whichResponse.split(
                            ' ')[1].upper() + '_rft_ref_depth.npz')
                        ref_depth = ref_depth_f[ref_depth_f.files[0]]
                        yFlow = np.array([])
                        interp = interpolate.interp1d(rft_depth, rft_prop, kind='linear', bounds_error=False,
                                                      fill_value=(rft_prop[0], rft_prop[-1]))
                        for d in ref_depth:
                            yFlow = np.append(yFlow, interp(d))
                    except:
                        yFlow = rft_prop
                else:
                    # If well, read the rsm file
                    if ext_data_info is not None:  # Get the data at a specific well and time
                        yFlow = self.ecl_case.summary_data(whichResponse, time)
            elif len(whichResponse.split(' ')) == 1:  # field data
                if whichResponse.upper() in ['FOPT', 'FWPT', 'FGPT', 'FWIT', 'FGIT']:
                    if ext_data_info is not None:
                        yFlow = self.ecl_case.summary_data(whichResponse, time)
                elif whichResponse.upper() in ['PERMX', 'PERMY', 'PERMZ', 'PORO', 'NTG', 'SATNUM',
                                               'MULTNUM', 'OPERNUM']:
                    yFlow = np.array([self.ecl_case.cell_data(whichResponse).flatten()[time]]) # assume that time is the index
                else:
                    yFlow = self.ecl_case.cell_data(whichResponse, time).flatten()
                    if yFlow is None:
                        yFlow = self.ecl_case.cell_data(whichResponse).flatten()

            # store the run time. NB: elapsed must be defined in .DATA file for this to work
            if 'save_elapsed' in self.input_dict and len(self.run_time) <= member:
                self.run_time.extend(self.ecl_case.summary_data('ELAPSED', time))

            # If we have performed coarsening, we store the number of active grid-cells
            if self.upscale is not None:
                # Get this number from INIT file
                with ecl.EclipseFile('En_' + str(member) + os.sep + self.file, 'INIT') as case:
                    intHead = case.get('INTEHEAD')
                # The active cell is element 12 of this vector, index 11 in python indexing...
                active_cells = intHead[11]
                if len(self.num_act) <= member:
                    self.num_act.extend([active_cells])

        else:
            case = ecl.EclipseCase(self.file + '.DATA')
            if ext_data_info[0] == 'days':
                time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                    dt.timedelta(days=ext_data_info[1])
            else:
                time = ext_data_info[1]

            # Check if the data is a field or well data, by checking if the well is defined
            if len(whichResponse.split(' ')) == 2:
                # if rft, search for rft_
                if 'rft_' in whichResponse:
                    rft_case = ecl.EclipseRFT(self.file)
                    # Get the data. Due to formating we can slice the property.
                    rft_prop = rft_case.rft_data(well=whichResponse.split(
                        ' ')[1], prop=whichResponse.split(' ')[0][4:])
                    # rft_data are collected for open connections. This may vary throughout the simulation, hence we
                    # must also collect the depth for the rft_data to check if all data is present
                    rft_depth = rft_case.rft_data(
                        well=whichResponse.split(' ')[1], prop='DEPTH')
                    try:
                        ref_depth_f = np.load(whichResponse.split(
                            ' ')[1].upper() + '_rft_ref_depth.npz')
                        ref_depth = ref_depth_f[ref_depth_f.files[0]]
                        yFlow = np.array([])
                        interp = interpolate.interp1d(rft_depth, rft_prop, kind='linear', bounds_error=False,
                                                      fill_value=(rft_prop[0], rft_prop[-1]))
                        for d in ref_depth:
                            yFlow = np.append(yFlow, interp(d))
                    except:
                        yFlow = rft_prop
                else:
                    # If well, read the rsm file
                    if ext_data_info is not None:  # Get the data at a specific well and time
                        yFlow = case.summary_data(whichResponse, time)
            elif len(whichResponse.split(' ')) == 1:
                if whichResponse in ['FOPT', 'FWPT', 'FGPT', 'FWIT', 'FGIT']:
                    if ext_data_info is not None:
                        yFlow = case.summary_data(whichResponse, time)
                else:
                    yFlow = case.cell_data(whichResponse, time).flatten()
                    if yFlow is None:
                        yFlow = case.cell_data(whichResponse).flatten()

            # If we have performed coarsening, we store the number of active grid-cells
            if self.upscale is not None:
                # Get this number from INIT file
                with ecl.EclipseFile('En_' + str(member) + os.sep + self.file,'INIT') as case:
                    intHead = case.get('INTEHEAD')
                # The active cell is element 12 of this vector, index 11 in python indexing...
                active_cells = intHead[11]
                if len(self.num_act) <= member:
                    self.num_act.extend([active_cells])

        return yFlow

    def store_fwd_debug(self, assimstep):
        if 'fwddebug' in self.keys_fwd:
            # Init dict. of variables to save
            save_dict = {}

            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(self.keys_fwd['fwddebug'], list):
                debug = self.keys_fwd['fwddebug']
            else:
                debug = [self.keys_fwd['fwddebug']]

            # Loop over variables to store in save list
            for var in debug:
                # Save with key equal variable name and the actual variable
                if isinstance(eval('self.' + var), dict):
                    # save directly
                    np.savez('fwd_debug_%s_%i' % (var, assimstep), **eval('self.' + var))
                else:
                    save_dict[var] = eval('self.' + var)

            np.savez('fwd_debug_%i' % (assimstep), **save_dict)

    def write_to_grid(self, value, propname, path, dim, t_ind=None):
        if t_ind == None:
            trans_dict = {}

            def _lookup(kw):
                return trans_dict[kw] if kw in trans_dict else kw

            # Write a quantity to the grid as a grdecl file
            with open(path + propname + '.grdecl', 'wb') as fileobj:
                grdecl._write_kw(fileobj, propname, value, _lookup, dim)
        else:
            pass
            # some errors with rips
            # p = Process(target=_write_to_resinsight, args=(list(value[~value.mask]),propname, t_ind))
            # Find an open resinsight case
            # p.start()
            # time.sleep(1)
            # p.terminate()

# def _write_to_resinsight(value, name,t_ind):
#     resinsight = rips.Instance.find()
#     case = resinsight.project.case(case_id=0)
#     case.set_active_cell_property(value, 'GENERATED', name,t_ind)


class ecl_100(eclipse):
    '''
    ecl_100 class
    '''

    def call_sim(self, path=None, wait_for_proc=False):
        """
        Method for calling the ecl_100 simulator.

        Parameters
        ----------
        path : str
            Alternative folder for the ecl_100.data file.

        wait_for_proc : bool, optional
            Logical variable to wait for the simulator to finish. Default is False.

        Returns
        -------
        .RSM : str
            Run summary file in the standard ECL format. Well data are collected from this file.

        .RST : str
            Restart file in the standard ECL format. Pressure and saturation data are collected from this file.

        .PRT : str
            Info file to be used for checking errors in the run.


        Changelog
        ---------
        - KF 14/9-2015
        """

        # Filename
        if path is not None:
            filename = path + self.file
        else:
            filename = self.file

        # Run the simulator:
        success = True
        try:
            with EclipseRunEnvironment(filename):
                com = ['eclrun', '--nocleanup', 'eclipse', filename + '.DATA']
                if 'sim_limit' in self.options:
                    call(com, stdout=DEVNULL, timeout=self.options['sim_limit'])
                else:
                    call(com, stdout=DEVNULL)
                raise ValueError
        except:
            print('\nError in the eclipse run.')  # add rerun?
            if not os.path.exists('Crashdump'):
                copytree(path, 'Crashdump')
            success = False

        return success


class ecl_300(eclipse):
    '''
    eclipse 300 class
    '''

    def call_sim(self, path=None, wait_for_proc=False):
        """
        Method for calling the ecl_300 simulator.

        Parameters
        ----------
        path : str
            Alternative folder for the ecl_100.data file.

        wait_for_proc : bool, optional
            Logical variable to wait for the simulator to finish. Default is False.

            !!! note
                For now, this option is only utilized in a single localization option.

        Returns
        -------
        RSM : str
            Run summary file in the standard ECL format. Well data are collected from this file.

        RST : str
            Restart file in the standard ECL format. Pressure and saturation data are collected from this file.

        PRT : str
            Info file to be used for checking errors in the run.

        Changelog
        ---------
        - KF 8/10-2015


        """
        # Filename
        if path is not None:
            filename = path + self.file
        else:
            filename = self.file

        # Run the simulator:
        with EclipseRunEnvironment(filename):
            call(['eclrun', '--nocleanup', 'e300', filename + '.DATA'], stdout=DEVNULL)
