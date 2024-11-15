"""Quality Assurance of the forecast (QA) and analysis (QC) step."""
import numpy as np
import os
# import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap
import itertools
import logging
from pipt.misc_tools import cov_regularization
from scipy.interpolate import interp1d
from scipy.io import loadmat
# import cv2


# Define the class for qa/qc tools.
class QAQC:
    """
     Perform Quality Assurance of the forecast (QA) and analysis (QC) step.
     Available functions (developed in 4DSEIS project and not available yet):

        - `calc_coverage`: check forecast data coverage
        - `calc_mahalanobis`: evaluate "higher-order" data coverage
        - `calc_kg`: check/write individual gain for parameters;
        -          flag data which have conflicting updates
        - `calc_da_stat`: compute statistics for updated parameters

     Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
     """

    # Initialize
    def __init__(self, keys, obs_data, datavar, logger=None, prior_info=None, sim=None, ini_state=None):
        self.keys = keys  # input info for the case
        self.obs_data = obs_data  # observed (real) data
        self.datavar = datavar  # data variance
        if logger is None:  # define a logger to print ouput
            logging.basicConfig(level=logging.INFO,
                                filename='qaqc_logger.log',
                                filemode='a',
                                format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
            self.logger = logging.getLogger('QAQC')
        else:
            self.logger = logging.getLogger('PET.PIPT.QCQA')
        self.prior_info = prior_info  # prior info for the different parameter types
        # this class contains potential writing functions (this class can be saved to debug_analysis)
        self.sim = sim
        self.ini_state = ini_state  # the first state; used to compute statistics
        self.ne = 0
        if self.ini_state is not None:
            # get the ensemble size from here
            self.ne = self.ini_state[list(self.ini_state.keys())[0]].shape[1]

        assim_step = 0  # Assume simultaneous assimiation
        assim_ind = [keys['obsname'], keys['assimindex'][assim_step]]
        if isinstance(assim_ind[1], list):  # Check if prim. ind. is a list
            self.l_prim = [int(x) for x in assim_ind[1]]
        else:  # Float
            self.l_prim = [int(assim_ind[1])]

        self.data_types = list(obs_data[0].keys())  # All data types
        self.en_obs = {}
        self.en_obs_vec = {}
        self.en_time = {}
        self.en_time_vec = {}
        for typ in self.data_types:
            self.en_obs[typ] = np.array(
                [self.obs_data[ind][typ].flatten() for ind in self.l_prim if self.obs_data[ind][typ]
                 is not None and self.obs_data[ind][typ].shape == (1,)])
            l = [self.obs_data[ind][typ].flatten() for ind in self.l_prim if self.obs_data[ind][typ] is not None
                 and self.obs_data[ind][typ].shape[0] > 1]
            if l:
                self.en_obs_vec[typ] = np.expand_dims(np.concatenate(l), 1)
            self.en_time[typ] = [ind for ind in self.l_prim if self.obs_data[ind][typ]
                                 is not None and self.obs_data[ind][typ].shape == (1,)]
            l = [ind for ind in self.l_prim if self.obs_data[ind][typ]
                 is not None and self.obs_data[ind][typ].shape[0] > 1]
            if l:
                self.en_time_vec[typ] = l

        # Check if the QA folder is generated
        self.folder = 'QAQC' + os.sep
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)  # if not generate

        if 'localization' in self.keys:
            self.localization = cov_regularization.localization(self.keys['localization'],
                                                                self.keys['truedataindex'],
                                                                self.keys['datatype'],
                                                                self.keys['staticvar'],
                                                                self.ne)
        self.pred_data = None
        self.state = None
        self.en_fcst = {}
        self.en_fcst_vec = {}
        self.lam = None

    # Set the predicted data and current state
    def set(self, pred_data, state=None, lam=None):
        self.pred_data = pred_data
        for typ in self.data_types:
            self.en_fcst[typ] = np.array(
                [self.pred_data[ind][typ].flatten() for ind in self.l_prim if self.obs_data[ind][typ]
                 is not None and self.obs_data[ind][typ].shape == (1,)])
            l = [self.pred_data[ind][typ] for ind in self.l_prim if self.obs_data[ind][typ] is not None
                 and self.obs_data[ind][typ].shape[0] > 1]
            if l:
                self.en_fcst_vec[typ] = np.concatenate(l)
        self.state = state
        self.lam = lam

    def calc_coverage(self, line=None):
        """
        Calculate the Data coverage for production and seismic data. For seismic data the plotting is based on the
        importance-scaled coverage developed by Espen O. Lie from GeoCore.

        Parameters
        ----------
        line : array-like, optional
            If not None, plot 1D coverage.

        Notes
        -----
        - Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        - Not available in current version of PIPT
        """

    def calc_kg(self, options=None):
        """
        Check/write individual gain for parameters.
        Note form ES gain with an identity Cd... This can be improved

        Visualization of the many of these parameters is problem-specific. In reservoir simulation cases, it is necessary
        to write this to the simulation grid. While for other applications, one might want other visualization. Hence,
        the method also depends on a simulator specific writer.

        Parameters
        ----------
        options : dict
            Settings for the Kalman gain computations.

            - 'num_store' : int, optional
                Number of elements to store. Default is 10.
            - 'unique_time' : bool, optional
                Calculate for each time instance. Default is False.
            - 'plot_all_kg' : bool, optional
                Plot all the Kalman gains for the field parameters. If False, plot the num_store. Default is False.
            - 'only_log' : bool, optional
                Only write to the logger; no plotting. Default is True.
            - 'auto_ada_loc' : bool, optional
                Use localization in computations. Default is True.
            - 'write_to_resinsight' : bool, optional
               Pipe results to ResInsight. Default is False.

                !!! note "requires that ResInsight is open on the computer."

        Notes
        -----
        - Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        - Not available in current version of PIPT
        """

    def calc_mahalanobis(self, combi_list=(1, None)):
        """
        Calculate the mahalanobis distance as described in "Oliver, D. S. (2020). Diagnosing reservoir model deficiency
        for model improvement. Journal of Petroleum Science and Engineering, 193(February).
        https://doi.org/10.1016/j.petrol.2020.107367"

        Parameters
        ----------
        combi_list : list
            List of levels and possible combinations of datatypes. The list must be given as a tuple with pairs:
            - level : int
                Defines which level. Default is 1.
            - combi_typ : str
                Defines how data are combined. Default is no combine.

        Notes
        -----
        - Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        - Not available in current version of PIPT
        """

    def calc_da_stat(self, options=None):
        """
        Calculate statistics for the updated parameters. The persentage of parameters that have updates larger than one,
        two and three standard deviations (calculated from the initial ensemble) are flagged.

        Parameters
        ----------
        options : dict
            Settings for statistics.
            write_to_file : bool, optional
                Whether to write results to a .grdecl file. Defaults to False.

        Notes
        -----
        - Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        - Not available in current version of PIPT
        """
