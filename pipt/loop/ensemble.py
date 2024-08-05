"""Descriptive description."""

# External import
import logging
import os.path

import numpy
import numpy as np
import sys
from copy import deepcopy, copy
from scipy.linalg import solve, cholesky
from scipy.spatial import distance
import itertools
from geostat.decomp import Cholesky

# Internal import
from ensemble.ensemble import Ensemble as PETEnsemble
import misc.read_input_csv as rcsv
from pipt.misc_tools import wavelet_tools as wt
from pipt.misc_tools import cov_regularization
import pipt.misc_tools.analysis_tools as at


class Ensemble(PETEnsemble):
    """
    Class for organizing/initializing misc. variables and simulator for an
    ensemble-based inversion run. Inherits the PET ensemble structure
    """

    def __init__(self, keys_da, keys_en, sim):
        """
        Parameters
        ----------
        keys_da : dict
            Options for the data assimilation class

            - daalg: spesification of the method, first the main type (e.g., "enrml"), then the solver (e.g., "gnenrml")
            - analysis: update flavour ("approx", "full" or "subspace")
            - energy: percent of singular values kept after SVD
            - obsvarsave: save the observations as a file (default false)
            - restart: restart optimization from a restart file (default false)
            - restartsave: save a restart file after each successful iteration (defalut false)
            - analysisdebug: specify which class variables to save to the result files
            - truedataindex: order of the simulated data (for timeseries this is points in time)
            - obsname: unit for truedataindex (for timeseries this is days or hours or seconds, etc.)
            - truedata: the data, e.g., provided as a .csv file
            - assimindex: index for the data that will be used for assimilation
            - datatype: list with the name of the datatypes
            - staticvar: name of the static variables
            - datavar: data variance, e.g., provided as a .csv file

        keys_en : dict
            Options for the ensemble class

            - ne: number of perturbations used to compute the gradient
            - state: name of state variables passed to the .mako file
            - prior_<name>: the prior information the state variables, including mean, variance and variable limits

        sim : callable
            The forward simulator (e.g. flow)
        """


        # do the initiallization of the PETensemble
        super(Ensemble, self).__init__(keys_en, sim)

        # set logger
        self.logger = logging.getLogger('PET.PIPT')

        # write initial information
        self.logger.info(f'Starting a {keys_da["daalg"][0]} run with the {keys_da["daalg"][1]} algorithm applying the '
                         f'{keys_da["analysis"]} update scheme with {keys_da["energy"]} Energy.')

        # Internalize PIPT dictionary
        if not hasattr(self, 'keys_da'):
            self.keys_da = keys_da
        if not hasattr(self, 'keys_en'):
            self.keys_en = keys_en

        if self.restart is False:
            # Init in _init_prediction_output (used in run_prediction)
            self.prediction = None
            self.temp_state = None  # temporary state saving
            self.cov_prior = None  # Prior cov. matrix
            self.sparse_info = None  # Init in _org_sparse_representation
            self.sparse_data = []  # List of the compression info
            self.data_rec = []  # List of reconstructed data
            self.scale_val = None  # Use to scale data

            # Prepare sparse representation
            if 'compress' in self.keys_da:
                self._org_sparse_representation()

            self._org_obs_data()
            self._org_data_var()

            # define projection for centring and scaling
            self.proj = (np.eye(self.ne) - (1 / self.ne) *
                         np.ones((self.ne, self.ne))) / np.sqrt(self.ne - 1)

            # If we have dynamic state variables, we allocate keys for them in 'state'. Since we do not know the size
            #  of the arrays of the dynamic variables, we only allocate an NE list to be filled in later (in
            # calc_forecast)
            if 'dynamicvar' in self.keys_da:
                dyn_var = self.keys_da['dynamicvar'] if isinstance(self.keys_da['dynamicvar'], list) else \
                    [self.keys_da['dynamicvar']]
                for name in dyn_var:
                    self.state[name] = [None] * self.ne

            # Option to store the dictionaries containing observed data and data variance
            if 'obsvarsave' in self.keys_da and self.keys_da['obsvarsave'] == 'yes':
                np.savez('obs_var', obs=self.obs_data, var=self.datavar)

            # Initialize localization
            if 'localization' in self.keys_da:
                self.localization = cov_regularization.localization(self.keys_da['localization'],
                                                                    self.keys_da['truedataindex'],
                                                                    self.keys_da['datatype'],
                                                                    self.keys_da['staticvar'],
                                                                    self.ne)
            # Initialize local analysis
            if 'localanalysis' in self.keys_da:
                self.local_analysis = at.init_local_analysis(
                    init=self.keys_da['localanalysis'], state=self.state.keys())

            self.pred_data = [{k: np.zeros((1, self.ne), dtype='float32') for k in self.keys_da['datatype']}
                              for _ in self.obs_data]

            self.cell_index = None  # default value for extracting states

    def check_assimindex_sequential(self):
        """
        Check if assim. indices is given as a 2D list as is needed in sequential updating. If not, make it a 2D list
        """
        # Check if ASSIMINDEX is a list. If not, make it a 2D list
        if not isinstance(self.keys_da['assimindex'], list):
            self.keys_da['assimindex'] = [[self.keys_da['assimindex']]]

        # If ASSIMINDEX is a 1D list (either given in as a single row or single column), we reshape to a 2D list
        elif not isinstance(self.keys_da['assimindex'][0], list):
            assimindex_temp = [None] * len(self.keys_da['assimindex'])

            for i in range(len(self.keys_da['assimindex'])):
                assimindex_temp[i] = [self.keys_da['assimindex'][i]]

            self.keys_da['assimindex'] = assimindex_temp

    def check_assimindex_simultaneous(self):
        """
        Check if assim. indices is given as a 1D list as is needed in simultaneous updating. If not, make it a 2D list
        with one row.
        """
        # Check if ASSIMINDEX is a list. If not, make it a 2D list with one row
        if not isinstance(self.keys_da['assimindex'], list):
            self.keys_da['assimindex'] = [[self.keys_da['assimindex']]]

        # Check if ASSIMINDEX is a 1D list. If true, make it a 2D list with one row
        elif not isinstance(self.keys_da['assimindex'][0], list):
            self.keys_da['assimindex'] = [self.keys_da['assimindex']]

        # If ASSIMINDEX is a 2D list, we reshape it to a 2D list with one row
        elif isinstance(self.keys_da['assimindex'][0], list):
            self.keys_da['assimindex'] = [
                [item for sublist in self.keys_da['assimindex'] for item in sublist]]

    def _org_obs_data(self):
        """
        Organize the input true observed data. The obs_data will be a list of length equal length of "TRUEDATAINDEX",
        and each entery in the list will be a dictionary with keys equal to the "DATATYPE".
        Also, the pred_data variable (predicted data or forward simulation) will be initialized here with the same
        structure as the obs_data variable.

        .. warning:: An "N/A" entry in "TRUEDATA" is treated as a None-entry; that is, there is NOT an observed data at this
        assimilation step.

        .. warning:: The array associated with the first string inputted in "TRUEDATAINDEX" is assumed to be the "main"
        index, that is, the length of this array will determine the length of the obs_data list! There arrays
        associated with the subsequent strings in "TRUEDATAINDEX" are then assumed to be a subset of the first
        string.  An example: the first string is SOURCE (e.g., sources in CSEM), where the array will be a list of numbering
        for the sources; and the second string is FREQ, where the array associated will be a list of frequencies.

        .. note:: It is assumed that the number of data associated with a subset is the same for each index in the subset.
        For example: If two frequencies are inputted in FREQ, then the number of data for one SOURCE index and one
        frequency is 1/2 of the total no. of data for that SOURCE index. If three frequencies are inputted, the number
        of data for one SOURCE index and one frequencies is 1/3 of the total no of data for that SOURCE index,
        and so on.
        """

        # # Check if keys_da['datatype'] is a string or list, and make it a list if single string is given
        # if isinstance(self.keys_da['datatype'], str):
        #     datatype = [self.keys_da['datatype']]
        # else:
        #     datatype = self.keys_da['datatype']
        #
        # # Extract primary indices from "TRUEDATAINDEX"
        # if isinstance(self.keys_da['truedataindex'], list):  # List of prim. ind
        #     true_prim = self.keys_da['truedataindex']
        # else:  # Float
        #     true_prim = [self.keys_da['truedataindex']]
        #
        # # Check if a csv file has been included as "TRUEDATAINDEX". If so, we read it and make a list,
        # if isinstance(self.keys_da['truedataindex'], str) and self.keys_da['truedataindex'].endswith('.csv'):
        #     with open(self.keys_da['truedataindex']) as csvfile:
        #         reader = csv.reader(csvfile)  # get a reader object
        #         true_prim = []  # Initialize the list of csv data
        #         for rows in reader:  # Rows is a list of values in the csv file
        #             csv_data = [None] * len(rows)
        #             for ind, col in enumerate(rows):
        #                 csv_data[ind] = int(col)
        #             true_prim.extend(csv_data)
        #     self.keys_da['truedataindex'] = true_prim
        #
        # # Check if a csv file has been included as "PREDICTION". If so, we read it and make a list,
        # if 'prediction' in self.keys_da:
        #     if isinstance(self.keys_da['prediction'], str) and self.keys_da['prediction'].endswith('.csv'):
        #         with open(self.keys_da['prediction']) as csvfile:
        #             reader = csv.reader(csvfile)  # get a reader object
        #             pred_prim = []  # Initialize the list of csv data
        #             for rows in reader:  # Rows is a list of values in the csv file
        #                 csv_data = [None] * len(rows)
        #                 for ind, col in enumerate(rows):
        #                     csv_data[ind] = int(col)
        #                 pred_prim.extend(csv_data)
        #         self.keys_da['prediction'] = pred_prim

        # Extract the observed data from "TRUEDATA"
        if len(self.keys_da['truedataindex']) == 1:  # Only one assimilation step
            if isinstance(self.keys_da['truedata'], list):
                truedata = [self.keys_da['truedata']]
            else:
                truedata = [[self.keys_da['truedata']]]
        else:  # More than one assim. step
            if isinstance(self.keys_da['truedata'][0], list):  # 2D list
                truedata = self.keys_da['truedata']
            else:
                truedata = [[x] for x in self.keys_da['truedata']]  # Make it a 2D list

        # Initialize obs_data list. List length = len("TRUEDATAINDEX"); dictionary in each list entry = d
        self.obs_data = [None] * len(self.keys_da['truedataindex'])

        # Check if a csv file has been included in TRUEDATA. If so, we read it and make a 2D list, which we can use
        # in the below when assigning data to obs_data dictionary
        if isinstance(self.keys_da['truedata'], str) and self.keys_da['truedata'].endswith('.csv'):
            truedata = rcsv.read_data_csv(
                self.keys_da['truedata'], self.keys_da['datatype'], self.keys_da['truedataindex'])

        # # Check if assimindex is given as a csv file. If so, we read and make a potential 2D list (if sequential).
        # if isinstance(self.keys_da['assimindex'], str) and self.keys_da['assimindex'].endswith('.csv'):
        #     with open(self.keys_da['assimindex']) as csvfile:
        #         reader = csv.reader(csvfile)  # get a reader object
        #         assimindx = []  # Initialize the 2D list of csv data
        #         for rows in reader:  # Rows is a list of values in the csv file
        #             csv_data = [None] * len(rows)
        #             for col in range(len(rows)):
        #                 csv_data[col] = int(rows[col])
        #             assimindx.append(csv_data)
        #     self.keys_da['assimindex'] = assimindx

        # Now we loop over all list entries in obs_data and fill in the observed data from "TRUEDATA".
        # NOTE: Not all data types may have observed data at each "TRUEDATAINDEX"; in this case it will have a None
        # entry.
        # NOTE2: If "TRUEDATA" contains a .npz file, this will be loaded. BUT the array loaded MUST be a 1D numpy
        # array! So resize BEFORE saving the .npz file!
        # NOTE3: If CSV file has been included in TRUEDATA, we read the data from this file
        vintage = 0
        for i in range(len(self.obs_data)):  # TRUEDATAINDEX
            # Init. dict. with datatypes (do inside loop to avoid copy of same entry)
            self.obs_data[i] = {}
            # Make unified inputs
            if 'unif_in' in self.keys_da and self.keys_da['unif_in'] == 'yes':
                if isinstance(truedata[i][0], str) and truedata[i][0].endswith('.npz'):
                    load_data = np.load(truedata[i][0])  # Load the .npz file
                    data_array = load_data[load_data.files[0]]

                    # Perform compression if required (we only and always compress signals with same size as number of active cells)
                    if self.sparse_info is not None and \
                            vintage < len(self.sparse_info['mask']) and \
                            len(data_array) == int(np.sum(self.sparse_info['mask'][vintage])):
                        data_array = self.compress(data_array, vintage, False)
                        vintage = vintage + 1

                    # Save array in obs_data. If it is an array with single value (not list), then we convert it to a
                    # list with one entry.
                    self.obs_data[i][self.keys_da['datatype'][0]] = np.array(
                        [data_array[()]]) if data_array.shape == () else data_array

                    # Entry is N/A, i.e., no data given
                elif isinstance(truedata[i][0], str) and not truedata[i][0].endswith('.npz') \
                        and truedata[i][0].lower() == 'n/a':
                    self.obs_data[i][self.keys_da['datatype'][0]] = None

                # Unknown string entry
                elif isinstance(truedata[i][0], str) and not truedata[i][0].endswith('.npz') \
                        and not truedata[i][0].lower() == 'n/a':
                    print(
                        '\n\033[1;31mERROR: Cannot load observed data file! Maybe it is not a .npz file?\033[1;m')
                    sys.exit(1)
                # Entry is a numerical value
                elif not isinstance(truedata[i][0], str):  # Some numerical value or None
                    self.obs_data[i][self.keys_da['datatype'][0]] = np.array(
                        truedata[i][:])  # no need to make this into a list
            else:
                for j in range(len(self.keys_da['datatype'])):  # DATATYPE
                    # Load a Numpy npz file
                    if isinstance(truedata[i][j], str) and truedata[i][j].endswith('.npz'):
                        load_data = np.load(truedata[i][j])  # Load the .npz file
                        data_array = load_data[load_data.files[0]]

                        # Perform compression if required (we only and always compress signals with same size as number of active cells)
                        if self.sparse_info is not None and \
                                vintage < len(self.sparse_info['mask']) and \
                                len(data_array) == int(np.sum(self.sparse_info['mask'][vintage])):
                            data_array = self.compress(data_array, vintage, False)
                            vintage = vintage + 1

                        # Save array in obs_data. If it is an array with single value (not list), then we convert it to a
                        # list with one entry
                        self.obs_data[i][self.keys_da['datatype'][j]] = np.array(
                            [data_array[()]]) if data_array.shape == () else data_array

                    # Entry is N/A, i.e., no data given
                    elif isinstance(truedata[i][j], str) and not truedata[i][j].endswith('.npz') \
                            and truedata[i][j].lower() == 'n/a':
                        self.obs_data[i][self.keys_da['datatype'][j]] = None

                    # Unknown string entry
                    elif isinstance(truedata[i][j], str) and not truedata[i][j].endswith('.npz') \
                            and not truedata[i][j].lower() == 'n/a':
                        print(
                            '\n\033[1;31mERROR: Cannot load observed data file! Maybe it is not a .npz file?\033[1;m')
                        sys.exit(1)

                    # Entry is a numerical value
                    # Some numerical value or None
                    elif not isinstance(truedata[i][j], str):
                        if type(truedata[i][j]) is numpy.ndarray:
                            self.obs_data[i][self.keys_da['datatype'][j]] = truedata[i][j]
                        else:
                            self.obs_data[i][self.keys_da['datatype'][j]] = np.array([truedata[i][j]])

                    # Scale data if required (currently only one group of data can be scaled)
                    if 'scale' in self.keys_da and self.keys_da['scale'][0] in self.keys_da['datatype'][j] and \
                            self.obs_data[i][self.keys_da['datatype'][j]] is not None:
                        self.obs_data[i][self.keys_da['datatype']
                                         [j]] *= self.keys_da['scale'][1]

    def _org_data_var(self):
        """
        Organize the input data variance given by the keyword "DATAVAR" in the "DATAASSIM" part the init_file.

        If a diagonal auto-covariance is to be used to generate data, there are two options for data variance: absolute
        and relative variance. Absolute is a fixed value for the variance, and relative is a percentage of
        the observed data as standard deviation which in turn is set as variance. If we want to use an empirical data
        covariance matrix to generate data, the user must supply a Numpy save file with samples, which is loaded here.
        If we want to specify the whole covariance matrix, this can also be done. The user must supply a Numpy save file
        which is loaded here.

        .. warning:: When relative variance is given as input, we set the variance as (true_obs_data*rel_perc*0.01)**2
        BECAUSE we often want this alternative in cases where we "add some percentage of Gaussian noise to the
        observed data". Hence, we actually want some percentage of the true observed data as STANDARD DEVIATION since
        it ultimately is the standard deviation (through square-root decompostion of Cd) that is used when adding
        noise to observed data.Note that this is ONLY a matter of definition, but we feel that this way of defining
        relative variance is most common.
        """
        # TODO: Change when sub-assim. indices have been re-implemented.

        # Check if keys_da['datatype'] is a string or list, and make it a list if single string is given
        if isinstance(self.keys_da['datatype'], str):
            datatype = [self.keys_da['datatype']]
        else:
            datatype = self.keys_da['datatype']

        # Extract primary indices from "TRUEDATAINDEX"
        if isinstance(self.keys_da['truedataindex'], list):  # List of prim. ind
            true_prim = self.keys_da['truedataindex']
        else:  # Float
            true_prim = [self.keys_da['truedataindex']]

        #
        # Extract the data variance from "DATAVAR"
        #
        # Only one assimilation step
        if len(true_prim) == 1:
            # More than one DATATYPE, but only one entry in DATAVAR
            if len(self.keys_da['datavar']) == 2 and len(datatype) > 1:
                # Copy list entry no. data type times
                datavar = [self.keys_da['datavar'] * len(datatype)]

            # One DATATYPE
            else:
                datavar = [self.keys_da['datavar']]

        # More than one assim. step
        else:
            # More than one DATATYPE, but only one entry in DATAVAR
            if not isinstance(self.keys_da['datavar'][0], list) and len(self.keys_da['datavar']) == 2 and \
                    len(datatype) > 1:
                # Need to make a list with entries equal to 2*no. data types (since there are 2 entries in DATAVAR
                # for one data type). Then we copy this list as many times as we have TRUEDATAINDEX (i.e.,
                # we get a 2D list)
                # Copy list entry no. data types times
                datavar_temp = self.keys_da['datavar'] * len(datatype)
                datavar = [None] * len(true_prim)  # Init.
                for i in range(len(true_prim)):
                    datavar[i] = deepcopy(datavar_temp)

            # Entry for each DATATYPE, but not for each TRUEDATAINDEX
            elif (len(self.keys_da['datavar'])) / 2 == len(datatype) and \
                    not isinstance(self.keys_da['datavar'][0], list):
                # If we have entry for each DATATYPE but NOT for each TRUEDATAINDEX, then we just copy the list of
                # entries to each TRUEDATAINDEX
                datavar = [None] * len(true_prim)  # Init.
                for i in range(len(true_prim)):
                    datavar[i] = deepcopy(self.keys_da['datavar'])

            else:
                datavar = self.keys_da['datavar']

        # Check if a csv file has been included in DATAVAR. If so datavar will be redefined and variance info will be
        #  extracted from the csv file
        if isinstance(self.keys_da['datavar'], str) and self.keys_da['datavar'].endswith('.csv'):
            datavar = rcsv.read_var_csv(self.keys_da['datavar'], datatype, true_prim)

        # Initialize datavar output
        self.datavar = [None] * len(true_prim)

        # Loop over all entries in datavar and fill in values from "DATAVAR" (use obs_data values in the REL variance
        #  cases)
        # TODO: Implement loading of data variance from .npz file
        vintage = 0
        for i in range(len(self.obs_data)):  # TRUEDATAINDEX
            # Init. dict. with datatypes (do inside loop to avoid copy of same entry)
            self.datavar[i] = {}
            for j in range(len(datatype)):  # DATATYPE
                # ABS
                # Absolute var.
                if datavar[i][2*j] == 'abs' and self.obs_data[i][datatype[j]] is not None:
                    self.datavar[i][datatype[j]] = datavar[i][2*j+1] * \
                        np.ones(len(self.obs_data[i][datatype[j]]))

                # REL
                # Rel. var.
                elif datavar[i][2*j] == 'rel' and self.obs_data[i][datatype[j]] is not None:
                    # Rel. var WITH a min. variance tolerance
                    if isinstance(datavar[i][2*j+1], list):
                        self.datavar[i][datatype[j]] = (datavar[i][2*j+1][0] * 0.01 *
                                                        self.obs_data[i][datatype[j]]) ** 2
                        ind_tol = self.datavar[i][datatype[j]] < datavar[i][2*j+1][1] ** 2
                        self.datavar[i][datatype[j]][ind_tol] = datavar[i][2*j+1][1] ** 2

                    else:  # Single. rel. var input
                        var = (datavar[i][2*j+1] * 0.01 * self.obs_data[i][datatype[j]]) ** 2
                        var = np.clip(var, 1.0e-9, None)  # avoid zero variance
                        self.datavar[i][datatype[j]] = var
                # EMP
                elif datavar[i][2*j] == 'emp' and datavar[i][2*j+1].endswith('.npz') and \
                        self.obs_data[i][datatype[j]] is not None:  # Empirical var.
                    load_data = np.load(datavar[i][2*j+1])  # load the numpy savez file
                    # store in datavar
                    self.datavar[i][datatype[j]] = load_data[load_data.files[0]]

                # LOAD
                elif datavar[i][2*j] == 'load' and datavar[i][2*j+1].endswith('.npz') and \
                        self.obs_data[i][datatype[j]] is not None:  # Load variance. (1d array)
                    load_data = np.load(datavar[i][2*j+1])  # load the numpy savez file
                    load_data = load_data[load_data.files[0]]
                    self.datavar[i][datatype[j]] = load_data  # store in datavar

                # CD the full covariance matrix is given in its correct format. Hence, load once and set as CD
                elif datavar[i][2 * j] == 'cd' and datavar[i][2 * j + 1].endswith('.npz') and \
                        self.obs_data[i][datatype[j]] is not None:
                    if not hasattr(self, 'cov_data'):  # check to populate once
                        # load the numpy savez file
                        load_data = np.load(datavar[i][2 * j + 1])
                        self.cov_data = load_data[load_data.files[0]]
                    # store the variance
                    self.datavar[i][datatype[j]] = self.cov_data[i*j, i*j]

                elif self.obs_data[i][datatype[j]] is None:  # No observed data
                    self.datavar[i][datatype[j]] = None  # Set None type here also

                # Handle case when noise is estimated using wavelets
                if self.sparse_info is not None and self.datavar[i][datatype[j]] is not None and \
                        vintage < len(self.sparse_info['mask']) and \
                        len(self.datavar[i][datatype[j]]) == int(np.sum(self.sparse_info['mask'][vintage])):
                    # compute var from sparse_data
                    est_noise = np.power(self.sparse_data[vintage].est_noise, 2)
                    self.datavar[i][datatype[j]] = est_noise  # override the given value
                    vintage = vintage + 1

    def _org_sparse_representation(self):
        """
        Function for reading input to wavelet sparse representation of data.
        """
        self.sparse_info = {}
        parsed_info = self.keys_da['compress']
        dim = [int(elem) for elem in parsed_info[0][1]]
        # flip to align with flow / eclipse
        self.sparse_info['dim'] = [dim[2], dim[1], dim[0]]
        self.sparse_info['mask'] = []
        for vint in range(1, len(parsed_info[1])):
            if not os.path.exists(parsed_info[1][vint]):
                mask = np.ones(self.sparse_info['dim'], dtype=bool)
                np.savez(f'mask_{vint-1}.npz', mask=mask)
            else:
                mask = np.load(parsed_info[1][vint])['mask']
            self.sparse_info['mask'].append(mask.flatten())
        self.sparse_info['level'] = parsed_info[2][1]
        self.sparse_info['wname'] = parsed_info[3][1]
        self.sparse_info['colored_noise'] = True if parsed_info[4][1] == 'yes' else False
        self.sparse_info['threshold_rule'] = parsed_info[5][1]
        self.sparse_info['th_mult'] = parsed_info[6][1]
        self.sparse_info['use_hard_th'] = True if parsed_info[7][1] == 'yes' else False
        self.sparse_info['keep_ca'] = True if parsed_info[8][1] == 'yes' else False
        self.sparse_info['inactive_value'] = parsed_info[9][1]
        self.sparse_info['use_ensemble'] = True if parsed_info[10][1] == 'yes' else None
        self.sparse_info['order'] = parsed_info[11][1]
        self.sparse_info['min_noise'] = parsed_info[12][1]

    def _ext_obs(self):
        self.obs_data_vector, _ = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                       self.list_datatypes)
        # Generate the data auto-covariance matrix
        if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
            if hasattr(self, 'cov_data'):  # cd matrix has been imported
                tmp_E = np.dot(cholesky(self.cov_data).T,
                               np.random.randn(self.cov_data.shape[0], self.ne))
            else:
                tmp_E = at.extract_tot_empirical_cov(
                    self.datavar, self.assim_index, self.list_datatypes, self.ne)
            # self.E = (tmp_E - tmp_E.mean(1)[:,np.newaxis])/np.sqrt(self.ne - 1)/
            if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                tmp_E = at.screen_data(tmp_E, self.aug_pred_data,
                                       self.obs_data_vector, self.iteration)
            self.E = tmp_E
            self.real_obs_data = self.obs_data_vector[:, np.newaxis] - tmp_E

            self.cov_data = np.var(self.E, ddof=1,
                                   axis=1)  # calculate the variance, to be used for e.g. data misfit calc
            # self.cov_data = ((self.E * self.E)/(self.ne-1)).sum(axis=1) # calculate the variance, to be used for e.g. data misfit calc
            self.scale_data = np.sqrt(self.cov_data)
        else:
            if not hasattr(self, 'cov_data'):  # if cd is not loaded
                self.cov_data = at.gen_covdata(
                    self.datavar, self.assim_index, self.list_datatypes)
            # data screening
            if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                self.cov_data = at.screen_data(
                    self.cov_data, self.aug_pred_data, self.obs_data_vector, self.iteration)

            init_en = Cholesky()  # Initialize GeoStat class for generating realizations
            self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector, self.cov_data, self.ne,
                                                                   return_chol=True)

    def _ext_state(self):
        # get vector of scaling
        self.state_scaling = at.calc_scaling(
            self.prior_state, self.list_states, self.prior_info)

        delta_scaled_prior = self.state_scaling[:, None] * \
            np.dot(at.aug_state(self.prior_state, self.list_states), self.proj)

        u_d, s_d, v_d = np.linalg.svd(delta_scaled_prior, full_matrices=False)

        # remove the last singular value/vector. This is because numpy returns all ne values, while the last is actually
        # zero. This part is a good place to include eventual additional truncation.
        energy = 0
        trunc_index = len(s_d) - 1  # inititallize
        for c, elem in enumerate(s_d):
            energy += elem
            if energy / sum(s_d) >= self.trunc_energy:
                trunc_index = c  # take the index where all energy is preserved
                break
        u_d, s_d, v_d = u_d[:, :trunc_index +
                            1], s_d[:trunc_index + 1], v_d[:trunc_index + 1, :]
        self.Am = np.dot(u_d, np.eye(trunc_index+1) *
                         ((s_d**(-1))[:, None]))  # notation from paper

    def save_temp_state_assim(self, ind_save):
        """
        Method to save the state variable during the assimilation. It is stored in a list with length = tot. no.
        assim. steps + 1 (for the init. ensemble). The list of temporary states are also stored as a .npz file.

        Parameters
        ----------
        ind_save : int
            Assim. step to save (0 = prior)
        """
        # Init. temp. save
        if ind_save == 0:
            # +1 due to init. ensemble
            self.temp_state = [None]*(len(self.get_list_assim_steps()) + 1)

        # Save the state
        self.temp_state[ind_save] = deepcopy(self.state)
        np.savez('temp_state_assim', self.temp_state)

    def save_temp_state_iter(self, ind_save, max_iter):
        """
        Save a snapshot of state at current iteration. It is stored in a list with length equal to max. iteration
        length + 1 (due to prior state being 0). The list of temporary states are also stored as a .npz file.

        .. warning:: Max. iterations must be defined before invoking this method.

        Parameters
        ----------
        ind_save : int
            Iteration step to save (0 = prior)
        """
        # Initial save
        if ind_save == 0:
            self.temp_state = [None] * (int(max_iter) + 1)  # +1 due to init. ensemble

        # Save state
        self.temp_state[ind_save] = deepcopy(self.state)
        np.savez('temp_state_iter', self.temp_state)

    def save_temp_state_mda(self, ind_save):
        """
        Save a snapshot of the state during a MDA loop. The temporary state will be stored as a list with length
        equal to the tot. no. of assimilations + 1 (init. ensemble saved in 0 entry). The list of temporary states
        are also stored as a .npz file.

        .. warning:: Tot. no. of assimilations must be defined before invoking this method.

        Parameter
        ---------
        ind_save : int
            Assim. step to save (0 = prior)
        """
        # Initial save
        if ind_save == 0:
            # +1 due to init. ensemble
            self.temp_state = [None] * (int(self.tot_assim) + 1)

        # Save state
        self.temp_state[ind_save] = deepcopy(self.state)
        np.savez('temp_state_mda', self.temp_state)

    def save_temp_state_ml(self, ind_save):
        """
        Save a snapshot of the state during a ML loop. The temporary state will be stored as a list with length
        equal to the tot. no. of assimilations + 1 (init. ensemble saved in 0 entry). The list of temporary states
        are also stored as a .npz file.

        .. warning:: Tot. no. of assimilations must be defined before invoking this method.

        Parameters
        ----------
        ind_save : int
            Assim. step to save (0 = prior)
        """
        # Initial save
        if ind_save == 0:
            # +1 due to init. ensemble
            self.temp_state = [None] * (int(self.tot_assim) + 1)

        # Save state
        self.temp_state[ind_save] = deepcopy(self.state)
        np.savez('temp_state_ml', self.temp_state)

    def compress(self, data=None, vintage=0, aug_coeff=None):
        """
        Compress the input data using wavelets.

        Parameters
        ----------
        data:
            data to be compressed
            If data is `None`, all data (true and simulated) is re-compressed (used if leading indices are updated)
        vintage: int
            the time index for the data
        aug_coeff: bool
            - False: in this case the leading indices for wavelet coefficients are computed
            - True: in this case the leading indices are augmented using information from the ensemble
            - None: in this case simulated data is compressed
        """

        # If input data is None, we re-compress all data
        data_array = None
        if data is None:
            vintage = 0
            for i in range(len(self.obs_data)):  # TRUEDATAINDEX
                for j in self.obs_data[i].keys():  # DATATYPE

                    data_array = self.obs_data[i][j]

                    # Perform compression if required
                    if data_array is not None and \
                            vintage < len(self.sparse_info['mask']) and \
                            len(data_array) == int(np.sum(self.sparse_info['mask'][vintage])):
                        data_array, wdec_rec = self.sparse_data[vintage].compress(
                            data_array)  # compress
                        self.obs_data[i][j] = data_array  # save array in obs_data
                        rec = self.sparse_data[vintage].reconstruct(
                            wdec_rec)  # reconstruct the data
                        s = 'truedata_rec_' + str(vintage) + '.npz'
                        np.savez(s, rec)  # save reconstructed data
                        est_noise = np.power(self.sparse_data[vintage].est_noise, 2)
                        self.datavar[i][j] = est_noise

                        # Update the ensemble
                        data_sim = self.pred_data[i][j]
                        self.pred_data[i][j] = np.zeros((len(data_array), self.ne))
                        self.data_rec.append([])
                        for m in range(self.pred_data[i][j].shape[1]):
                            data_array = data_sim[:, m]
                            data_array, wdec_rec = self.sparse_data[vintage].compress(
                                data_array)  # compress
                            self.pred_data[i][j][:, m] = data_array
                            rec = self.sparse_data[vintage].reconstruct(
                                wdec_rec)  # reconstruct the data
                            self.data_rec[vintage].append(rec)

                        # Go to next vintage
                        vintage = vintage + 1

            # Option to store the dictionaries containing observed data and data variance
            if 'obsvarsave' in self.keys_da and self.keys_da['obsvarsave'] == 'yes':
                np.savez('obs_var', obs=self.obs_data, var=self.datavar)

            if 'saveforecast' in self.keys_en:
                s = 'prior_forecast_rec.npz'
                np.savez(s, self.data_rec)

            data_array = None

        elif aug_coeff is None:

            data_array, wdec_rec = self.sparse_data[vintage].compress(data)
            rec = self.sparse_data[vintage].reconstruct(
                wdec_rec)  # reconstruct the simulated data
            if len(self.data_rec) == vintage:
                self.data_rec.append([])
            self.data_rec[vintage].append(rec)

        elif not aug_coeff:

            options = copy(self.sparse_info)
            # find the correct mask for the vintage
            options['mask'] = options['mask'][vintage]
            if type(options['min_noise']) == list:
                if 0 <= vintage < len(options['min_noise']):
                    options['min_noise'] = options['min_noise'][vintage]
                else:
                    print(
                        'Error: min_noise must either be scalar or list with one number for each vintage')
                    sys.exit(1)
            x = wt.SparseRepresentation(options)
            data_array, wdec_rec = x.compress(data, self.sparse_info['th_mult'])
            self.sparse_data.append(x)  # store the information
            data_rec = x.reconstruct(wdec_rec)  # reconstruct the data
            s = 'truedata_rec_' + str(vintage) + '.npz'
            np.savez(s, data_rec)  # save reconstructed data
            if self.sparse_info['use_ensemble']:
                data_array = data  # just return the same as input

        elif aug_coeff:

            _, _ = self.sparse_data[vintage].compress(data, self.sparse_info['th_mult'])
            data_array = data  # just return the same as input

        return data_array

    def local_analysis_update(self):
        '''
        Function for updates that can be used by all algorithms. Do this once to avoid duplicate code for local
        analysis.
        '''
        orig_list_data = deepcopy(self.list_datatypes)
        orig_list_state = deepcopy(self.list_states)
        orig_cd = deepcopy(self.cov_data)
        orig_real_obs_data = deepcopy(self.real_obs_data)
        orig_data_vector = deepcopy(self.obs_data_vector)
        # loop over the states that we want to update. Assume that the state and data combinations have been
        # determined by the initialization.
        # TODO: augment parameters with identical mask.
        for state in self.local_analysis['region_parameter']:
            self.list_datatypes = [elem for elem in self.list_datatypes if
                                   elem in self.local_analysis['update_mask'][state]]
            self.list_states = [deepcopy(state)]
            self._ext_state()  # scaling for this state
            if 'localization' in self.keys_da:
                self.localization.loc_info['field'] = self.state_scaling.shape
            del self.cov_data
            # reset the random state for consistency
            np.random.set_state(self.data_random_state)
            self._ext_obs()  # get the data that's in the list of data.
            _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                         self.list_datatypes)
            # Mean pred_data and perturbation matrix with scaling
            if len(self.scale_data.shape) == 1:
                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
            else:
                self.pert_preddata = solve(
                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

            aug_state = at.aug_state(self.current_state, self.list_states)
            self.update()
            if hasattr(self, 'step'):
                aug_state_upd = aug_state + self.step
            self.state = at.update_state(aug_state_upd, self.state, self.list_states)

        for state in self.local_analysis['vector_region_parameter']:
            current_list_datatypes = deepcopy(self.list_datatypes)
            for state_indx in range(self.state[state].shape[0]): # loop over the elements in the region
                self.list_datatypes = [elem for elem in self.list_datatypes if
                                       elem in self.local_analysis['update_mask'][state][state_indx]]
                if len(self.list_datatypes):
                    self.list_states = [deepcopy(state)]
                    self._ext_state()  # scaling for this state
                    if 'localization' in self.keys_da:
                        self.localization.loc_info['field'] = self.state_scaling.shape
                    del self.cov_data
                    # reset the random state for consistency
                    np.random.set_state(self.data_random_state)
                    self._ext_obs()  # get the data that's in the list of data.
                    _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                                 self.list_datatypes)
                    # Mean pred_data and perturbation matrix with scaling
                    if len(self.scale_data.shape) == 1:
                        self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                    np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
                    else:
                        self.pert_preddata = solve(
                            self.scale_data, np.dot(self.aug_pred_data, self.proj))

                    aug_state = at.aug_state(self.current_state, self.list_states)[state_indx,:]
                    self.update()
                    if hasattr(self, 'step'):
                        aug_state_upd = aug_state + self.step[state_indx,:]
                    self.state[state][state_indx,:] = aug_state_upd

                self.list_datatypes = deepcopy(current_list_datatypes)

        for state in self.local_analysis['cell_parameter']:
            self.list_states = [deepcopy(state)]
            self._ext_state()  # scaling for this state
            orig_state_scaling = deepcopy(self.state_scaling)
            param_position = self.local_analysis['parameter_position'][state]
            field_size = param_position.shape
            for k in range(field_size[0]):
                for j in range(field_size[1]):
                    for i in range(field_size[2]):
                        current_data_list = list(
                            self.local_analysis['update_mask'][state][k][j][i])
                        current_data_list.sort()  # ensure consistent ordering of data
                        if len(current_data_list):
                            # if non-unique data for assimilation index, get the relevant data.
                            if self.local_analysis['unique'] == False:
                                orig_assim_index = deepcopy(self.assim_index)
                                assim_index_data_list = set(
                                    [el.split('_')[0] for el in current_data_list])
                                current_assim_index = [
                                    int(el.split('_')[1]) for el in current_data_list]
                                current_data_list = list(assim_index_data_list)
                                self.assim_index[1] = current_assim_index
                            self.list_datatypes = deepcopy(current_data_list)
                            del self.cov_data
                            # reset the random state for consistency
                            np.random.set_state(self.data_random_state)
                            self._ext_obs()
                            _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data,
                                                                         self.assim_index,
                                                                         self.list_datatypes)
                            # get parameter indexes
                            full_cell_index = np.ravel_multi_index(
                                np.array([[k], [j], [i]]), tuple(field_size))
                            # count active values
                            self.cell_index = [sum(param_position.flatten()[:el])
                                               for el in full_cell_index]
                            if 'localization' in self.keys_da:
                                self.localization.loc_info['field'] = (
                                    len(self.cell_index),)
                                self.localization.loc_info['distance'] = cov_regularization._calc_distance(
                                    self.local_analysis['data_position'],
                                    self.local_analysis['unique'],
                                    current_data_list, self.assim_index,
                                    self.obs_data, self.pred_data, [(k, j, i)])
                            # Set relevant state scaling
                            self.state_scaling = orig_state_scaling[self.cell_index]

                            # Mean pred_data and perturbation matrix with scaling
                            if len(self.scale_data.shape) == 1:
                                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data,
                                                                                            self.proj)
                            else:
                                self.pert_preddata = solve(
                                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

                            aug_state = at.aug_state(
                                self.current_state, self.list_states, self.cell_index)
                            self.update()
                            if hasattr(self, 'step'):
                                aug_state_upd = aug_state + self.step
                            self.state = at.update_state(
                                aug_state_upd, self.state, self.list_states, self.cell_index)

                            if self.local_analysis['unique'] == False:
                                # reset assim index
                                self.assim_index = deepcopy(orig_assim_index)
                            if hasattr(self, 'localization') and 'distance' in self.localization.loc_info:  # reset
                                del self.localization.loc_info['distance']

        self.list_datatypes = deepcopy(orig_list_data)  # reset to original list
        self.list_states = deepcopy(orig_list_state)
        self.cov_data = deepcopy(orig_cd)
        self.real_obs_data = deepcopy(orig_real_obs_data)
        self.obs_data_vector = deepcopy(orig_data_vector)
        self.cell_index = None
