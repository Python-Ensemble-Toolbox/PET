# External imports
import os
import numpy as np
import logging
import time
import pickle

# Gets or creates a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # set log level
file_handler = logging.FileHandler('popt.log')  # define file handler and set formatter
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
    variable) will be iterated upon using an algorithm defined in the update_scheme package.

    Attributes
    ----------
        logger : Logger
            Print output to screen and log-file

        pickle_restart_file : str
            Save name for pickle dump/load

        optimize_result : OptimizeResult
            Dictionary with results for the current iteration

        iteration : int
             Iteration index

        max_iter : int
            Max number of iterations

        restart : bool
            Restart flag

        restartsave : bool
            Save restart information flag

    Methods
    -------
        run_loop()
            The main optimization loop

        save()
            Save restart file

        load()
            Load restart file

        calc_update()
            Empty dummy function, actual functionality must be defined by the subclasses

    """

    def __init__(self, **options):
        """
        Parameters
        ----------
            options: dict
                Optimization options
        """

        def __set__variable(var_name=None, defalut=None):
            if var_name in options:
                return options[var_name]
            else:
                return defalut

        # Set the logger
        self.logger = logger

        # Save name for (potential) pickle dump/load
        self.pickle_restart_file = 'popt_restart_dump'

        # Dictionary with results for the current iteration
        self.optimize_result = None

        # Initial iteration index
        self.iteration = 0

        # Time counter and random generator
        self.start_time = None
        self.rnd = None

        # Max number of iterations
        self.max_iter = __set__variable('maxiter', 10)

        # Restart flag
        self.restart = __set__variable('restart', False)

        # Save restart information flag
        self.restartsave = __set__variable('restartsave', False)

        # Initialize options, state variable and function values
        self.options = None
        self.mean_state = None
        self.obj_func_values = None

        # Initialize number of function and jacobi evaluations
        self.nfev = 0
        self.njev = 0

    def run_loop(self):
        """
        This is the main optimization loop.
        """

        # If it is a restart run, we load the self info that exists in the pickle save file.
        if self.restart:

            # Check if the pickle save file exists in folder
            assert (self.pickle_restart_file in [f for f in os.listdir('.') if os.path.isfile(f)]), \
                'The restart file "{0}" does not exist in folder. Cannot restart!'.format(self.pickle_restart_file)

            # Load restart file
            self.load()

            # Set the random generator to be the saved value
            np.random.set_state(self.rnd)

        else:

            # delete potential restart files to avoid any problems
            if self.pickle_restart_file in [f for f in os.listdir('.') if os.path.isfile(f)]:
                os.remove(self.pickle_restart_file)

            self.iteration += 1

        # Run a while loop until max iterations or convergence is reached
        is_successful = True
        while self.iteration <= self.max_iter and is_successful:

            # Update control variable
            is_successful = self.calc_update()

            # Save restart file (if requested)
            if self.restartsave:
                self.rnd = np.random.get_state()  # get the current random state
                self.save()

        # Check if max iterations was reached
        if self.iteration > self.max_iter:
            self.optimize_result['message'] = 'Iterations stopped due to max iterations reached!'
        else:
            self.optimize_result['message'] = 'Convergence was met :)'

        # Logging some info to screen
        logger.info('       Optimization converged in %d iterations ', self.iteration-1)
        logger.info('       Optimization converged with final obj_func = %.4f',
                    np.mean(self.optimize_result['fun']))
        logger.info('       Total number of function evaluations = %d', self.optimize_result['nfev'])
        logger.info('       Total number of jacobi evaluations = %d', self.optimize_result['njev'])
        if self.start_time is not None:
            logger.info('       Total elapsed time = %.2f minutes', (time.perf_counter()-self.start_time)/60)
        logger.info('       ============================================')

    def save(self):
        """
        We use pickle to dump all the information we have in 'self'. Can be used, e.g., if some error has occurred.
        """
        # Open save file and dump all info. in self
        with open(self.pickle_restart_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        """
        Load a pickled file and save all info. in self.
        """
        # Open file and read with pickle
        with open(self.pickle_restart_file, 'rb') as f:
            tmp_load = pickle.load(f)

        # Save in 'self'
        self.__dict__.update(tmp_load)

    def calc_update(self):
        """
        This is an empty dummy function. Actual functionality must be defined by the subclasses.
        """
        pass
