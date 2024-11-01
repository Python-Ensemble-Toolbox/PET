# External imports
import numpy as np
from numpy import linalg as la
import time
import pprint

# Internal imports
from popt.nisc_tools import optim_tools as ot
from popt.loop.optimize import Optimize
import popt.update_schemes.optimizers as opt


class EnOpt(Optimize):
    r"""
    This is an implementation of the ensemble steepest descent ensemble optimization algorithm - EnOpt.
    The update of the control variable is done with the simple steepest (or gradient) descent algorithm:

    .. math::
        x_l = x_{l-1} - \alpha \times C \times G

    where :math:`x` is the control variable, :math:`l` is the iteration index, :math:`\alpha` is the step size,
    :math:`C` is a smoothing matrix (e.g., covariance matrix for :math:`x`), and :math:`G` is the ensemble gradient.

    Methods
    -------
    calc_update()
        Update using steepest descent method with ensemble gradient

    References
    ----------
    Chen et al., 2009, 'Efficient Ensemble-Based Closed-Loop Production Optimization', SPE Journal, 14 (4): 634-645.
    """

    def __init__(self, fun, x, args, jac, hess, bounds=None, **options):

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

        hess: callable
            Hessian function

        bounds: list, optional
            (min, max) pairs for each element in x. None is used to specify no bound.

        options: dict
            Optimization options

                - maxiter: maximum number of iterations (default 10)
                - restart: restart optimization from a restart file (default false)
                - restartsave: save a restart file after each successful iteration (defalut false)
                - tol: convergence tolerance for the objective function (default 1e-6)
                - alpha: step size for the steepest decent method (default 0.1)
                - beta: momentum coefficient for running accelerated optimization (default 0.0)
                - alpha_maxiter: maximum number of backtracing trials (default 5)
                - resample: number indicating how many times resampling is tried if no improvement is found
                - optimizer: 'GA' (gradient accent) or Adam (default 'GA')
                - nesterov: use Nesterov acceleration if true (default false)
                - hessian: use Hessian approximation (if the algorithm permits use of Hessian) (default false)
                - normalize: normalize the gradient if true (default true)
                - cov_factor: factor used to shrink the covariance for each resampling trial (defalut 0.5)
                - savedata: specify which class variables to save to the result files (state, objective
                            function value, iteration number, number of function evaluations, and number
                            of gradient evaluations, are always saved)
        """

        # init PETEnsemble
        super(EnOpt, self).__init__(**options)

        def __set__variable(var_name=None, defalut=None):
            if var_name in options:
                return options[var_name]
            else:
                return defalut

        # Set input as class variables
        self.options = options  # options
        self.fun = fun  # objective function
        self.cov = args[0]  # initial covariance
        self.jac = jac  # gradient function
        self.hess = hess  # hessian function
        self.bounds = bounds  # parameter bounds
        self.mean_state = x  # initial mean state

        # Set other optimization parameters
        self.obj_func_tol = __set__variable('tol', 1e-6)
        self.alpha = __set__variable('alpha', 0.1)
        self.alpha_cov = __set__variable('alpha_cov', 0.001)
        self.beta = __set__variable('beta', 0.0)  # this is stored in the optimizer class
        self.nesterov = __set__variable('nesterov', False)  # use Nesterov acceleration if value is true
        self.alpha_iter_max = __set__variable('alpha_maxiter', 5)
        self.max_resample = __set__variable('resample', 0)
        self.use_hessian = __set__variable('hessian', False)
        self.normalize = __set__variable('normalize', True)
        self.cov_factor = __set__variable('cov_factor', 0.5)

        # Initialize other variables
        self.state_step = 0  # state step
        self.cov_step = 0  # covariance step

        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()
            self.obj_func_values = self.fun(self.mean_state, **self.epf)
            self.nfev += 1
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)
            if self.logger is not None:
                self.logger.info('\n\n')
                self.logger.info('       ====== Running optimization - EnOpt ======')
                self.logger.info('\n'+pprint.pformat(self.options))
                info_str = '       {:<10} {:<10} {:<15} {:<15} {:<15} '.format('iter', 'alpha_iter',
                                                                        'obj_func', 'step-size', 'cov[0,0]')
                self.logger.info(info_str)
                self.logger.info('       {:<21} {:<15.4e}'.format(self.iteration, np.mean(self.obj_func_values)))

        # Initialize optimizer
        optimizer = __set__variable('optimizer', 'GA')
        if optimizer == 'GA':
            self.optimizer = opt.GradientAscent(self.alpha, self.beta)
        elif optimizer == 'Adam':
            self.optimizer = opt.Adam(self.alpha, self.beta)
        elif optimizer == 'AdaMax':
            self.normalize = False
            self.optimizer = opt.AdaMax(self.alpha, self.beta)

        # The EnOpt class self-ignites, and it is possible to send the EnOpt class as a callale method to scipy.minimize
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
                                    shrink*(self.cov + self.beta*self.cov_step), **self.epf)
            else:
                gradient = self.jac(self.mean_state, shrink*self.cov, **self.epf)
            self.njev += 1

            # Compute the hessian
            hessian = self.hess()
            if self.use_hessian:
                inv_hessian = np.linalg.inv(hessian)
                gradient = inv_hessian @ (self.cov @ self.cov) @ gradient
                if self.normalize:
                    hessian /= np.maximum(la.norm(hessian, np.inf), 1e-12)  # scale the hessian with inf-norm
            elif self.normalize:
                gradient /= np.maximum(la.norm(gradient, np.inf), 1e-12)  # scale the gradient with inf-norm
                hessian /= np.maximum(la.norm(hessian, np.inf), 1e-12)  # scale the hessian with inf-norm

            # Initialize for this step
            alpha_iter = 0

            while improvement is False:  # backtracking loop

                new_state, new_step = self.optimizer.apply_update(self.mean_state, gradient, iter=self.iteration)
                new_state = ot.clip_state(new_state, self.bounds)

                # Calculate new objective function
                new_func_values = self.fun(new_state, **self.epf)
                self.nfev += 1

                if np.mean(self.obj_func_values) - np.mean(new_func_values) > self.obj_func_tol:

                    # Update objective function values and state
                    self.obj_func_values = new_func_values
                    self.mean_state = new_state
                    self.state_step = new_step
                    self.alpha = self.optimizer.get_step_size()

                    # Update covariance (currently we don't apply backtracking for alpha_cov)
                    self.cov_step = self.alpha_cov * hessian + self.beta * self.cov_step
                    self.cov = self.cov - self.cov_step
                    self.cov = ot.get_sym_pos_semidef(self.cov)

                    # Write logging info
                    if self.logger is not None:
                        info_str_iter = '       {:<10} {:<10} {:<15.4e} {:<15.2e} {:<15.2e}'.\
                            format(self.iteration, alpha_iter, np.mean(self.obj_func_values),
                                    self.alpha, self.cov[0, 0])
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




    
        
