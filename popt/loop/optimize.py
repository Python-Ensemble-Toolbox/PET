# External imports
import os
import numpy as np
import time
import pickle
from abc import ABC, abstractmethod

# Internal imports
import popt.misc_tools.optim_tools as ot
from ensemble.logger import PetLogger


class Optimize(ABC):
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
        options : dict
            Optimization options
        """
        # Setup logger
        self.logger = PetLogger('optim.log')

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
        self.max_iter = options.get('maxiter', 20)

        # Restart flag
        self.restart = options.get('restart', False)

        # Save restart information flag
        self.restartsave = options.get('restartsave', False)

        # Optimze with external penalty function for constraints, provide r_0 as input
        self.epf = options.get('epf', {})
        self.epf_iteration = 0

        # Initialize variables (set in subclasses)
        self.options = None
        self.mean_state = None
        self.obj_func_values = None
        self.obj_func_tol = None  # objective tolerance limit

        # Initialize number of function and jacobi evaluations
        self.nfev = 0
        self.njev = 0

        self.msg = 'Convergence was met :)'

    # Abstract function that subclasses are forced to define
    @abstractmethod
    def fun(self, x, *args, **kwargs):  # objective function
        pass

    # Abstract properties that subclasses are forced to define
    @property
    @abstractmethod
    def xk(self):  # current state
        pass

    @property
    @abstractmethod
    def ftol(self):  # function tolerance
        pass

    @ftol.setter
    @abstractmethod
    def ftol(self, value):  # setter for function tolerance
        pass

    def run_loop(self):
        """
        This is the main optimization loop.
        """

        # If it is a restart run, we load the self info that exists in the pickle save file.
        if self.restart:
            try:
                self.load()
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                raise RuntimeError(f"Failed to load restart file '{self.pickle_restart_file}': {e}")
            # Set the random generator to be the saved value
            np.random.set_state(self.rnd)
        else:
            # delete potential restart files to avoid any problems
            if self.pickle_restart_file in [f for f in os.listdir('.') if os.path.isfile(f)]:
                os.remove(self.pickle_restart_file)
            self.iteration += 1

        # Check if external penalty function (epf) for handling constraints should be used
        epf_not_converged = True
        previous_state = None
        if self.epf:
            previous_state = self.mean_state
            self.logger(
                f'─────> EPF-EnOpt: {self.epf_iteration}, {self.epf["r"]} (outer iteration, penalty factor)'
            )  # print epf info

        while epf_not_converged:  # outer loop using epf

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
            if self.iteration >= self.max_iter:
                self.msg = 'Optimization stopped due to maximum iterations reached!'
                self.optimize_result['message'] = self.msg
            else:
                if not isinstance(self.msg, str): self.msg = ''
                self.optimize_result['message'] = self.msg

            # Logging some info to screen
            self.logger('')
            self.logger('============================================')
            self.logger(self.msg)
            self.logger(f'Optimization converged in {self.iteration-1} iterations ')
            self.logger(f'Optimization converged with final obj_func = {np.mean(self.optimize_result["fun"]):.4f}')
            self.logger(f'Total number of function evaluations = {self.optimize_result["nfev"]}')
            self.logger(f'Total number of jacobi evaluations = {self.optimize_result["njev"]}')
            if self.start_time is not None:
                self.logger(f'Total elapsed time = {(time.perf_counter()-self.start_time)/60:.2f} minutes')
            self.logger('============================================')

            # Test for convergence of outer epf loop
            epf_not_converged = False
            if self.epf:
                if self.epf_iteration > self.epf['max_epf_iter']:  # max epf_iterations set to 10
                    self.logger(f'─────> EPF-EnOpt: maximum epf iterations reached')  # print epf info
                    break
                p = np.abs(previous_state-self.mean_state) / (np.abs(previous_state) + 1.0e-9)
                conv_crit = self.epf['conv_crit']
                if np.any(p > conv_crit):
                    epf_not_converged = True
                    previous_state = self.mean_state
                    self.epf['r'] *= self.epf['r_factor']  # increase penalty factor
                    self.obj_func_tol *= self.epf['tol_factor']  # decrease tolerance
                    self.obj_func_values = self.fun(self.mean_state, **self.epf)
                    self.iteration = 0
                    self.epf_iteration += 1
                    optimize_result = ot.get_optimize_result(self)
                    ot.save_optimize_results(optimize_result)
                    self.nfev += 1
                    self.iteration = +1
                    r = self.epf['r']
                    self.logger(f'─────> EPF-EnOpt: {self.epf_iteration}, {r} (outer iteration, penalty factor)')  # print epf info
                else:
                    self.logger(f'─────> EPF-EnOpt: converged, no variables changed more than {conv_crit*100} %')  # print epf info
                    final_obj_no_penalty = str(round(float(self.fun(self.mean_state)),4))
                    self.logger(f'─────> EPF-EnOpt: objective value without penalty = {final_obj_no_penalty}') # print epf info
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
