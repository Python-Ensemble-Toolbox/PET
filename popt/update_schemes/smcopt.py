"""Stochastic Monte-Carlo optimisation."""
# External imports
import numpy as np
import time
import pprint

# Internal imports
from popt.loop.optimize import Optimize
import popt.update_schemes.optimizers as opt
from popt.misc_tools import optim_tools as ot


class SmcOpt(Optimize):
    """
    TODO: Write docstring ala EnOpt
    """

    def __init__(self, fun, x, args, sens, bounds=None, **options):
        """
        Parameters
        ----------
        fun : callable
            objective function

        x : ndarray
            Initial state

        sens : callable
            Ensemble sensitivity

        bounds : list, optional
            (min, max) pairs for each element in x. None is used to specify no bound.

        options : dict
            Optimization options

            - maxiter: maximum number of iterations (default 10)
            - restart: restart optimization from a restart file (default false)
            - restartsave: save a restart file after each successful iteration (defalut false)
            - tol: convergence tolerance for the objective function (default 1e-6)
            - alpha: weight between previous and new step (default 0.1)
            - alpha_maxiter: maximum number of backtracing trials (default 5)
            - resample: number indicating how many times resampling is tried if no improvement is found
            - cov_factor: factor used to shrink the covariance for each resampling trial (defalut 0.5)
            - inflation_factor: term used to weight down prior influence (defalult 1)
            - survival_factor: fraction of surviving samples
            - savedata: specify which class variables to save to the result files (state, objective function
                        value, iteration number, number of function evaluations, and number of gradient
                        evaluations, are always saved)
        """

        # init PETEnsemble
        super(SmcOpt, self).__init__(**options)

        def __set__variable(var_name=None, defalut=None):
            if var_name in options:
                return options[var_name]
            else:
                return defalut

        # Set input as class variables
        self.options = options  # options
        self.fun = fun  # objective function
        self.sens = sens  # gradient function
        self.bounds = bounds  # parameter bounds
        self.mean_state = x  # initial mean state
        self.best_state = None  # best ensemble member
        self.cov = args[0]  # covariance matrix for sampling

        # Set other optimization parameters
        self.obj_func_tol = __set__variable('tol', 1e-6)
        self.alpha = __set__variable('alpha', 0.1)
        self.alpha_iter_max = __set__variable('alpha_maxiter', 5)
        self.max_resample = __set__variable('resample', 0)
        self.cov_factor = __set__variable('cov_factor', 0.5)
        self.inflation_factor = __set__variable('inflation_factor', 1)
        self.survival_factor = __set__variable('survival_factor', 0)

        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()
            self.obj_func_values = self.fun(self.mean_state)
            self.best_func = np.mean(self.obj_func_values)
            self.nfev += 1
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)
            if self.logger is not None:
                self.logger.info('       ====== Running optimization - SmcOpt ======')
                self.logger.info('\n' + pprint.pformat(self.options))
                info_str = '       {:<10} {:<10} {:<15} {:<15} '.format('iter', 'alpha_iter',
                                                                           'obj_func', 'step-size')
                self.logger.info(info_str)
                self.logger.info('       {:<21} {:<15.4e}'.format(self.iteration, np.mean(self.obj_func_values)))

        self.optimizer = opt.GradientAscent(self.alpha, 0)

        # The SmcOpt class self-ignites
        self.run_loop()  # run_loop resides in the Optimization class (super)

    def calc_update(self,):
        """
        Update using sequential monte carlo method
        """

        improvement = False
        success = False
        resampling_iter = 0
        inflate = 2 * (self.inflation_factor + self.iteration)

        while improvement is False:  # resampling loop

            # Shrink covariance each time we try resampling
            shrink = self.cov_factor ** resampling_iter

            # Calc sensitivity
            (sens_matrix, self.best_state, best_func_tmp) = self.sens(self.mean_state, inflate,
                                                                      shrink*self.cov, self.survival_factor)
            self.njev += 1

            # Initialize for this step
            alpha_iter = 0

            while improvement is False:  # backtracking loop

                search_direction = sens_matrix
                new_state = self.optimizer.apply_smc_update(self.mean_state, search_direction, iter=self.iteration)
                new_state = ot.clip_state(new_state, self.bounds)

                # Calculate new objective function
                new_func_values = self.fun(new_state)
                self.nfev += 1

                if np.mean(self.obj_func_values) - np.mean(new_func_values) > self.obj_func_tol or \
                          (self.best_func - best_func_tmp) > self.obj_func_tol:

                    # Update objective function values and step
                    self.obj_func_values = new_func_values
                    self.mean_state = new_state
                    if (self.best_func - best_func_tmp) > self.obj_func_tol:
                        self.best_func = best_func_tmp

                    # Write logging info
                    if self.logger is not None:
                        info_str_iter = '       {:<10} {:<10} {:<15.4e} {:<15.2e}'. \
                            format(self.iteration, alpha_iter, self.best_func,
                                   self.alpha)
                        self.logger.info(info_str_iter)

                    # Iteration was a success
                    improvement = True
                    success = True
                    self.optimizer.restore_parameters()

                    # Save variables defined in savedata keyword.
                    self.optimize_result = ot.get_optimize_result(self)
                    ot.save_optimize_results(self.optimize_result)

                    # Update iteration counter if iteration was successful and save current state
                    self.iteration += 1

                else:

                    # If we do not have a reduction in the objective function, we reduce the step limiter
                    if alpha_iter < self.alpha_iter_max:
                        self.optimizer.apply_backtracking()  # decrease alpha
                        alpha_iter += 1
                    elif (resampling_iter < self.max_resample and
                          np.mean(new_func_values) - np.mean(self.obj_func_values) > 0):  # update gradient
                        resampling_iter += 1
                        self.optimizer.restore_parameters()
                        break
                    else:
                        success = False
                        return success

        return success
