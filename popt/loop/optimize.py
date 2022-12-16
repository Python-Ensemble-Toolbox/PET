"""
Build class for running ensemble based optimization.
"""

import os
import sys
from numpy import savez
from numpy import linalg as la
import numpy as np
import logging
import time
import pickle
from copy import deepcopy

# Internal imports
from popt.misc_tools import basic_tools as bt

# Gets or creates a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # set log level
file_handler = logging.FileHandler('log_optim_loops.log')  # define file handler and set formatter
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # add file handler to logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Optimize:
    """
    Class for ensemble optimization algorithms. These are classified by calculating the sensitivity or gradient using
    ensemble instead of classical derivatives. The loop is else as a classic optimization loop: a state (or control
    variable) will be iterated upon using a algorithm defined in the update_scheme package.
    """

    def __init__(self, method):
        """

        """

        # Save name for (potential) pickle dump/load
        self.pickle_restart_file = 'emergency_dump'

        # If it is a restart run, we do not need to initialize anything, only load the self info that exists in the
        # pickle save file. If it is not a restart run, we initialize everything below.
        if 'restart' in method.keys_opt and method.keys_opt['restart'] == 'yes':

            # Check if the pickle save file exists in folder
            assert (self.pickle_restart_file in [f for f in os.listdir('..') if os.path.isfile(f)]), \
                'The restart file "{0}" does not exist in folder. Cannot restart!'.format(self.pickle_restart_file)

            # Load restart file
            self.load()

        else:

            # Internalize ensemble and simulator class instances
            self.method = method

            # Default max. iter (can be overwritten by user in a update_schemes method)
            self._ext_max_iter()

            # Initial iteration index
            self.iteration = 0

    def _ext_max_iter(self):
        """
        Extract max. iter from ENOPT keyword in OPTIM part if inputted.

        ST 4/5-18
        """
        # Default value for max. iterations
        default_max_iter = 50

        # Check if ENOPT has been given in OPTIM. If it is not present, we assign a default value
        if 'enopt' not in self.method.keys_opt:
            # Default value
            print('ENOPT keyword not found. Assigning default value to EnOpt parameters!')
        else:
            # Make ENOPT a 2D list
            if not isinstance(self.method.keys_opt['enopt'][0], list):
                enopt = [self.method.keys_opt['enopt']]
            else:
                enopt = self.method.keys_opt['enopt']

            # Assign ENOPT for MAX_ITER option. If MAX_ITER is not present, assign default value
            ind_max_iter = bt.index2d(enopt, 'max_iter')
            # MAX_ITER does not exist
            if None in ind_max_iter:
                self.max_iter = default_max_iter
                # MAX_ITER present; assign value
            else:
                self.max_iter = enopt[ind_max_iter[0]][ind_max_iter[1]+1]

    def run_loop(self):
        """
        The general optimization loop
        """

        # Logging output to screen, logger saved to log files.
        start_time = time.perf_counter()
        logger.info('Running optimization loops...')
        info_str = '{:<10} {:<10} {:<10} {:<10} {:<10}'.format('iter', 'alpha_iter', 'obj_func', 'alpha', 'cov')
        logger.info(info_str)
        info_str_iter = '{:<10} {:<10} {:<10.2f} {:<10.2e} {:<10.2e}'.\
            format(self.iteration, 0, np.mean(self.method.obj_func_values), 0, 0)
        logger.info(info_str_iter)
        self.iteration += 1

        # Run a while loop until max. iterations or convergence is reached
        is_successful = True
        while self.iteration <= self.max_iter and is_successful:

            # Update control variable
            is_successful = self.method.calc_update(self.iteration, logger)

            # Update iteration counter if iteration was successful
            if is_successful is True:
                self.iteration += 1
                np.savez('opt_state',self.method.state)  # save current state

            # Save restart file (if requested)
            if 'restartsave' in self.method.keys_opt and self.method.keys_opt['restartsave'] == 'yes':
                self.save()

        # Check if max iterations was reached
        if self.iteration > self.max_iter:
            reason = 'Iterations stopped due to max iterations reached!'
        else:
            reason = 'Convergence was met :)'

        # Save results
        np.savez('final_results', state=self.method.state, obj_func_values=self.method.obj_func_values,
                 total_iter=self.iteration-1, num_func_eval=self.method.num_func_eval, status=reason)

        # Logging some info to screen
        logger.info('Optimization converged in %d iterations ', self.iteration-1)
        logger.info('Optimization converged with final obj_func = %.2f', np.mean(self.method.obj_func_values))
        logger.info('Total number of function evaluations = %d', self.method.num_func_eval)
        logger.info('Total elapsed time = %.2f minutes', (time.perf_counter()-start_time)/60)

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
