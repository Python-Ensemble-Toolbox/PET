# External import
import logging
import numpy as np
import sys
from copy import deepcopy

# Internal import
from ensemble.ensemble import Ensemble as PETEnsemble
import misc.read_input_csv as rcsv
from pipt.misc_tools import wavelet_tools as wt
from pipt.misc_tools import cov_regularization

class Ensemble(PETEnsemble):
    """
    Class for organizing/initializing misc. variables and simulator for an ensemble-based inversion run. Inherits the
    PET ensemble structure
    """
    def __init__(self,keys_da,keys_en,sim):

        # do the initiallization of the PETensemble
        super(Ensemble,self).__init__(keys_da,sim)

        #set logger
        self.logger = logging.getLogger('PET.PIPT')

        # write initial information
        self.logger.info(f'Starting a {keys_da["daalg"][0]} run with the {keys_da["daalg"][1]} algorithm applying the '
                         f'{keys_da["analysis"]} update scheme with {keys_da["energy"]} Energy.')

        # Internalize PIPT dictionary
        self.keys_da = keys_da

        if self.restart is False:
            self.prediction = None  # Init in _init_prediction_output (used in run_prediction)
            self.temp_state = None  # temporary state saving
            self.cov_prior = None  # Prior cov. matrix
            self.sparse_info = None  # Init in _org_sparse_representation
            self.sparse_data = []  # List of the compression info (sendt to flow_rock.py)
            self.data_rec = [0] * int(self.keys_da['ne'])  # List of reconstructed seismic data

            self._org_obs_data()
            self._org_data_var()

            # Prepare sparse representation
            if 'compress' in self.keys_da:
                self._org_sparse_representation()

            # define projection for centring and scaling
            self.proj = (np.eye(self.ne) - (1 / self.ne) * np.ones((self.ne, self.ne))) / np.sqrt(self.ne - 1)

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

            self.pred_data = [{k: np.zeros((1, self.ne), dtype='float32') for k in self.keys_en['datatype']}
                              for _ in self.obs_data]
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
            self.keys_da['assimindex'] = [[item for sublist in self.keys_da['assimindex'] for item in sublist]]

    def _org_obs_data(self):
        """
        Organize the input true observed data. The obs_data will be a list of length equal length of "TRUEDATAINDEX",
        and each entery in the list will be a dictionary with keys equal to the "DATATYPE".
        Also, the pred_data variable (predicted data or forward simulation) will be initialized here with the same
        structure as the obs_data variable.
        ----------------------------------------------------------------------------------------------------------------
        OBS: An "N/A" entry in "TRUEDATA" is treated as a None-entry; that is, there is NOT an observed data at this
        assimilation step.

        OBS2: The array associated with the first string inputted in "TRUEDATAINDEX" is assumed to be the "main"
        index, that is, the length of this array will determine the length of the obs_data list! There arrays
        associated with the subsequent strings in "TRUEDATAINDEX" are then assumed to be a subset of the first
        string.
        An example: the first string is SOURCE (e.g., sources in CSEM), where the array will be a list of numbering
        for the sources; and the second string is FREQ, where the array associated will be a list of frequencies.

        NB! It is assumed that the number of data associated with a subset is the same for each index in the subset.
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
            truedata = rcsv.read_data_csv(self.keys_da['truedata'], self.keys_da['datatype'], self.keys_da['truedataindex'])

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
        for i in range(len(self.obs_data)):  # TRUEDATAINDEX
            self.obs_data[i] = {}  # Init. dict. with datatypes (do inside loop to avoid copy of same entry)
            if 'unif_in' in self.keys_da and self.keys_da['unif_in'] == 'yes':  # Make unified inputs
                if isinstance(truedata[i][0], str) and truedata[i][0].endswith('.npz'):
                    load_data = np.load(truedata[i][0])  # Load the .npz file
                    data_array = load_data[load_data.files[0]]

                    # Perform compression if required (we only compress signals with same size as number of active cells)
                    if self.sparse_info is not None and len(data_array) == int(np.sum(self.sparse_info['actnum'])):
                        x = wt.SparseRepresentation(self.sparse_info)
                        data_array, wdec_rec = x.compress(data_array, self.sparse_info['th_mult'])
                        self.sparse_data.append(x)  # store the information
                        data_rec = x.reconstruct(wdec_rec)  # reconstruct the data
                        s = truedata[i][j][:-4] + '_rec' + truedata[i][j][-4:]
                        np.savez(s, data_rec)  # save reconstructed data

                    # Save array in obs_data. If it is an array with single value (not list), then we convert it to a
                    # list with one entry.
                    self.obs_data[i][self.keys_da['datatype'][0]] = np.array([data_array[()]]) if data_array.shape == () else data_array

                    # Entry is N/A, i.e., no data given
                elif isinstance(truedata[i][0], str) and not truedata[i][0].endswith('.npz') \
                        and truedata[i][0].lower() == 'n/a':
                    self.obs_data[i][self.keys_da['datatype'][0]] = None

                # Unknown string entry
                elif isinstance(truedata[i][0], str) and not truedata[i][0].endswith('.npz') \
                        and not truedata[i][0].lower() == 'n/a':
                    print('\n\033[1;31mERROR: Cannot load observed data file! Maybe it is not a .npz file?\033[1;m')
                    sys.exit(1)
                # Entry is a numerical value
                elif not isinstance(truedata[i][0], str):  # Some numerical value or None
                    self.obs_data[i][self.keys_da['datatype'][0]] = np.array(truedata[i][:])  # no need to make this into a list
            else:
                for j in range(len(self.keys_da['datatype'])):  # DATATYPE
                    # Load a Numpy npz file
                    if isinstance(truedata[i][j], str) and truedata[i][j].endswith('.npz'):
                        load_data = np.load(truedata[i][j])  # Load the .npz file
                        data_array = load_data[load_data.files[0]]

                        # Perform compression if required
                        if self.sparse_info is not None and len(data_array) == int(np.sum(self.sparse_info['actnum'])):
                            x = wt.SparseRepresentation(self.sparse_info)
                            data_array, wdec_rec = x.compress(data_array, self.sparse_info['th_mult'])
                            self.sparse_data.append(x)
                            data_rec = x.reconstruct(wdec_rec)  # reconstruct the data
                            s = truedata[i][j][:-4] + '_rec' + truedata[i][j][-4:]
                            np.savez(s, data_rec)  # save reconstructed data

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
                        print('\n\033[1;31mERROR: Cannot load observed data file! Maybe it is not a .npz file?\033[1;m')
                        sys.exit(1)

                    # Entry is a numerical value
                    elif not isinstance(truedata[i][j], str):  # Some numerical value or None
                        self.obs_data[i][self.keys_da['datatype'][j]] = np.array([truedata[i][j]])

                    # Scale data if required (currently only one group of data can be scaled)
                    if 'scale' in self.keys_da and self.keys_da['scale'][0] in self.keys_da['datatype'][j] and \
                            self.obs_data[i][self.keys_da['datatype'][j]] is not None:
                        self.obs_data[i][self.keys_da['datatype'][j]] *= self.keys_da['scale'][1]

    def _org_data_var(self):
        """
        Organize the input data variance given by the keyword "DATAVAR" in the "DATAASSIM" part the init_file.

        If a diagonal auto-covariance is to be used to generate data, there are two options for data variance: absolute
        and relative variance. Absolute is a fixed value for the variance, and relative is a percentage of
        the observed data as standard deviation which in turn is set as variance. If we want to use an empirical data
        covariance matrix to generate data, the user must supply a Numpy save file with samples, which is loaded here.
        If we want to specify the whole covariance matrix, this can also be done. The user must supply a Numpy save file
        which is loaded here.
        ----------------------------------------------------------------------------------------------------------------
        OBS: When relative variance is given as input, we set the variance as (true_obs_data*rel_perc*0.01)**2
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
                datavar = [self.keys_da['datavar'] * len(datatype)]  # Copy list entry no. data type times

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
                datavar_temp = self.keys_da['datavar'] * len(datatype)  # Copy list entry no. data types times
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
            datavar = rcsv.read_var_csv(self.keys_da['datavar'],datatype, true_prim)

        # Initialize datavar output
        self.datavar = [None] * len(true_prim)

        # Loop over all entries in datavar and fill in values from "DATAVAR" (use obs_data values in the REL variance
        #  cases)
        # TODO: Implement loading of data variance from .npz file
        sparse_index = 0
        for i in range(len(self.obs_data)):  # TRUEDATAINDEX
            self.datavar[i] = {}  # Init. dict. with datatypes (do inside loop to avoid copy of same entry)
            for j in range(len(datatype)):  # DATATYPE
                # ABS
                if datavar[i][2*j] == 'abs' and self.obs_data[i][datatype[j]] is not None:  # Absolute var.
                    self.datavar[i][datatype[j]] = datavar[i][2*j+1]*np.ones(len(self.obs_data[i][datatype[j]]))

                # REL
                elif datavar[i][2*j] == 'rel' and self.obs_data[i][datatype[j]] is not None:  # Rel. var.
                    # Rel. var WITH a min. variance tolerance
                    if isinstance(datavar[i][2*j+1], list):
                        self.datavar[i][datatype[j]] = (datavar[i][2*j+1][0] * 0.01 *
                                                        self.obs_data[i][datatype[j]]) ** 2
                        ind_tol = self.datavar[i][datatype[j]] < datavar[i][2*j+1][1] ** 2
                        self.datavar[i][datatype[j]][ind_tol] = datavar[i][2*j+1][1] ** 2

                    else:  # Single. rel. var input
                        self.datavar[i][datatype[j]] = (datavar[i][2*j+1] * 0.01 * self.obs_data[i][datatype[j]]) ** 2
                # EMP
                elif datavar[i][2*j] == 'emp' and datavar[i][2*j+1].endswith('.npz') and \
                        self.obs_data[i][datatype[j]] is not None:  # Empirical var.
                    load_data = np.load(datavar[i][2*j+1])  # load the numpy savez file
                    self.datavar[i][datatype[j]] = load_data[load_data.files[0]]  # store in datavar

                # LOAD
                elif datavar[i][2*j] == 'load' and datavar[i][2*j+1].endswith('.npz') and \
                        self.obs_data[i][datatype[j]] is not None:  # Load variance. (1d array)
                    load_data = np.load(datavar[i][2*j+1])  # load the numpy savez file
                    load_data = load_data[load_data.files[0]]
                    if self.sparse_info is not None and len(load_data) == int(np.sum(self.sparse_info['actnum'])):  # compute var from sparse_data
                        est_noise = self.sparse_data[sparse_index].est_noise
                        load_data = np.power(est_noise, 2)  # from std to var
                        sparse_index = sparse_index + 1
                    self.datavar[i][datatype[j]] = load_data  # store in datavar

                # CD the full covariance matrix is given in its correct format. Hence, load once and set as CD
                elif datavar[i][2 * j] == 'cd' and datavar[i][2 * j + 1].endswith('.npz') and \
                     self.obs_data[i][datatype[j]] is not None:
                    if not hasattr(self, 'cov_data'): # check to populate once
                        load_data = np.load(datavar[i][2 * j + 1])  # load the numpy savez file
                        self.cov_data = load_data[load_data.files[0]]
                    self.datavar[i][datatype[j]] = self.cov_data[i*j, i*j]  # store the variance

                elif self.obs_data[i][datatype[j]] is None:  # No observed data
                    self.datavar[i][datatype[j]] = None  # Set None type here also

    def _org_sparse_representation(self):
        """
        Function for reading input to wavelet sparse representation of data.
        """
        self.sparse_info = {}
        parsed_info = self.keys_da['compress']
        self.sparse_info['dim'] = [int(elem) for elem in parsed_info[0][1]]
        self.sparse_info['actnum'] = np.load(parsed_info[1][1])['actnum']
        self.sparse_info['level'] = parsed_info[2][1]
        self.sparse_info['wname'] = parsed_info[3][1]
        self.sparse_info['colored_noise'] = True if parsed_info[4][1] == 'yes' else False
        self.sparse_info['threshold_rule'] = parsed_info[5][1]
        self.sparse_info['th_mult'] = parsed_info[6][1]
        self.sparse_info['use_hard_th'] = True if parsed_info[7][1] == 'yes' else False
        self.sparse_info['keep_ca'] = True if parsed_info[8][1] == 'yes' else False
        self.sparse_info['inactive_value'] = parsed_info[9][1]

    def save_temp_state_assim(self, ind_save):
        """
        Method to save the state variable during the assimilation. It is stored in a list with length = tot. no.
        assim. steps + 1 (for the init. ensemble). The list of temporary states are also stored as a .npz file.

        Parameter:
        ---------
        ind_save : int
                   Assim. step to save (0 = prior)
        """
        # Init. temp. save
        if ind_save == 0:
            self.temp_state = [None]*(len(self.get_list_assim_steps()) + 1)  # +1 due to init. ensemble

        # Save the state
        self.temp_state[ind_save] = deepcopy(self.state)
        np.savez('temp_state_assim', self.temp_state)

    def save_temp_state_iter(self, ind_save, max_iter):
        """
        Save a snapshot of state at current iteration. It is stored in a list with length equal to max. iteration
        length + 1 (due to prior state being 0). The list of temporary states are also stored as a .npz file.
        OBS: Max. iterations must be defined before invoking this method.

        Parameter:
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
        OBS: Tot. no. of assimilations must be defined before invoking this method.

        Parameter
        ---------
        ind_save : int
                   Assim. step to save (0 = prior)
        """
        # Initial save
        if ind_save == 0:
            self.temp_state = [None] * (int(self.tot_assim) + 1)  # +1 due to init. ensemble

        # Save state
        self.temp_state[ind_save] = deepcopy(self.state)
        np.savez('temp_state_mda', self.temp_state)

    def save_temp_state_ml(self, ind_save):
        """
        Save a snapshot of the state during a ML loop. The temporary state will be stored as a list with length
        equal to the tot. no. of assimilations + 1 (init. ensemble saved in 0 entry). The list of temporary states
        are also stored as a .npz file.
        OBS: Tot. no. of assimilations must be defined before invoking this method.

        Parameter
        --------
        ind_save : int
                   Assim. step to save (0 = prior)
        """
        # Initial save
        if ind_save == 0:
            self.temp_state = [None] * (int(self.tot_assim) + 1)  # +1 due to init. ensemble

        # Save state
        self.temp_state[ind_save] = deepcopy(self.state)
        np.savez('temp_state_ml', self.temp_state)
