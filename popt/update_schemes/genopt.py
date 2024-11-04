"""Non-Gaussian generalisation of EnOpt."""
# External imports
import numpy as np
from numpy import linalg as la
import time

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize
import popt.update_schemes.optimizers as opt
from popt.update_schemes.cma import CMA


class GenOpt(Optimize):
    
    def __init__(self, fun, x, args, jac, jac_mut, corr_adapt=None, bounds=None, **options):

        """
        Parameters
        ----------
        fun: callable
            objective function

        x: ndarray
            Initial state

        args: tuple
            Initial covariance

        jac: callable
            Gradient function

        jac_mut: callable
            Mutation gradient function
        
        corr_adapt : callable
            Function for correalation matrix adaption

        bounds: list, optional
            (min, max) pairs for each element in x. None is used to specify no bound.

        options: dict
                Optimization options
        """

        # init PETEnsemble
        super(GenOpt, self).__init__(**options)
        
        def __set__variable(var_name=None, defalut=None):
            if var_name in options:
                return options[var_name]
            else:
                return defalut

        # Set input as class variables
        self.options    = options    # options
        self.fun        = fun        # objective function
        self.jac        = jac        # gradient function
        self.jac_mut    = jac_mut    # mutation function
        self.corr_adapt = corr_adapt # correlation adaption function
        self.bounds     = bounds     # parameter bounds
        self.mean_state = x          # initial mean state
        self.theta      = args[0]    # initial theta and correlation
        self.corr       = args[1]    # inital correlation

        # Set other optimization parameters
        self.obj_func_tol   = __set__variable('obj_func_tol', 1e-6)
        self.alpha          = __set__variable('alpha', 0.1)
        self.alpha_theta    = __set__variable('alpha_theta', 0.1)
        self.alpha_corr     = __set__variable('alpha_theta', 0.1)
        self.beta           = __set__variable('beta', 0.0)        # this is stored in the optimizer class
        self.nesterov       = __set__variable('nesterov', False)  # use Nesterov acceleration if value is true
        self.alpha_iter_max = __set__variable('alpha_maxiter', 5)
        self.max_resample   = __set__variable('resample', 0)
        self.normalize      = __set__variable('normalize', True)
        self.cov_factor     = __set__variable('cov_factor', 0.5)

        # Initialize other variables
        self.state_step = 0  # state step
        self.theta_step = 0  # covariance step

        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()
            self.obj_func_values = self.fun(self.mean_state)
            self.nfev += 1
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)
            if self.logger is not None:
                self.logger.info('       Running optimization...')
                info_str = '       {:<10} {:<10} {:<15} {:<15} {:<10} {:<10} {:<10} {:<10} '.format('iter', 
                                                                                                    'alpha_iter',
                                                                                                    'obj_func', 
                                                                                                    'step-size', 
                                                                                                    'alpha0', 
                                                                                                    'beta0',
                                                                                                    'max corr',
                                                                                                    'min_corr')
                self.logger.info(info_str)
                self.logger.info('       {:<21} {:<15.4e}'.format(self.iteration, 
                                                                  round(np.mean(self.obj_func_values),4)))

        # Initialize optimizer
        optimizer = __set__variable('optimizer', 'GA')
        if optimizer == 'GA':
            self.optimizer = opt.GradientAscent(self.alpha, self.beta)
        elif optimizer == 'Adam':
            self.optimizer = opt.Adam(self.alpha, self.beta)

        # The GenOpt class self-ignites, and it is possible to send the EnOpt class as a callale method to scipy.minimize
        self.run_loop()  # run_loop resides in the Optimization class (super)

    def calc_update(self):
        """
        Update using steepest descent method with ensemble gradients
        """

        # Initialize variables for this step
        improvement = False
        success = False
        resampling_iter = 0

        while improvement is False:  # resampling loop

            # Shrink covariance each time we try resampling
            shrink = self.cov_factor ** resampling_iter

            # Calculate gradient
            if self.nesterov:
                gradient = self.jac(self.mean_state + self.beta*self.state_step,
                                    self.theta + self.beta*self.theta_step, self.corr)
            else:
                gradient = self.jac(self.mean_state, self.theta, self.corr)
            self.njev += 1

            # Compute the mutation gradient
            gradient_theta, en_matrices = self.jac_mut(return_ensembles=True)
            if self.normalize:
                gradient /= np.maximum(la.norm(gradient, np.inf), 1e-12)              # scale the gradient with inf-norm
                gradient_theta /= np.maximum(la.norm(gradient_theta, np.inf), 1e-12)  # scale the mutation with inf-norm

            # Initialize for this step
            alpha_iter = 0

            while improvement is False:  # backtracking loop

                new_state, new_step = self.optimizer.apply_update(self.mean_state, gradient, iter=self.iteration)
                new_state = ot.clip_state(new_state, self.bounds)

                # Calculate new objective function
                new_func_values = self.fun(new_state)
                self.nfev += 1

                if np.mean(self.obj_func_values) - np.mean(new_func_values) > self.obj_func_tol:

                    # Update objective function values and state
                    self.obj_func_values = new_func_values
                    self.mean_state = new_state
                    self.state_step = new_step
                    self.alpha = self.optimizer.get_step_size()

                    # Update theta (currently we don't apply backtracking for theta)
                    self.theta_step = self.beta*self.theta_step - self.alpha_theta*gradient_theta
                    self.theta = self.theta + self.theta_step

                    # update correlation matrix
                    if isinstance(self.corr_adapt, CMA):
                        enZ = en_matrices['gaussian']
                        enJ = en_matrices['objective']
                        self.corr = self.corr_adapt(cov  = self.corr, 
                                                    step = new_step/self.alpha, 
                                                    X = enZ.T, 
                                                    J = np.squeeze(enJ))
                        
                    elif callable(self.corr_adapt):
                        self.corr = self.corr - self.alpha_corr*self.corr_adapt()
                    
                    # Write logging info
                    if self.logger is not None:
                        corr_max = round(np.max(self.corr-np.eye(self.corr.shape[0])), 3)
                        corr_min = round(np.min(self.corr), 3)
                        info_str_iter = '       {:<10} {:<10} {:<15.4e} {:<15} {:<10} {:<10} {:<10} {:<10}'.\
                                        format(self.iteration,
                                               alpha_iter, 
                                               round(np.mean(self.obj_func_values),4),
                                               self.alpha, 
                                               round(self.theta[0, 0],2), 
                                               round(self.theta[0, 1],2),
                                               corr_max, 
                                               corr_min)
                        
                        self.logger.info(info_str_iter)

                    # Update step size in the one-dimensional case
                    if new_state.size == 1:
                        self.optimizer.step_size /= 2

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
