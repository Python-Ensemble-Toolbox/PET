"""Gradient acceleration."""
import logging

import numpy as np

__all__ = ['GradientDescent', 'Adam', 'AdaMax', 'Steihaug', ]


log = logging.getLogger(__name__)


class GradientDescent:
    r"""
    A class for performing gradient descent optimization with momentum and backtracking.
    The gradient descent update equation with momentum is given by:

    $$ \begin{align}
        v_t &= \beta * v_{t-1} + \alpha * gradient \\
        x_t &= x_{t-1} - v_t
    \end{align} $$


    Attributes
    -----------------------------------------------------------------------------------
    step_size : float
        The initial step size provided during initialization.

    momentum : float
        The initial momentum factor provided during initialization.

    velocity : array_like
        Current velocity of the optimization process.

    temp_velocity : array_like
        Temporary velocity

    _step_size : float
        Private attribute for temporarily modifying step size.

    _momentum : float
        Private attribute for temporarily modifying momentum.

    Methods
    -----------------------------------------------------------------------------------
    apply_update(control, gradient, **kwargs):
        Apply a gradient update to the control parameter.

    apply_backtracking() :
        Apply backtracking by reducing step size and momentum temporarily.

    restore_parameters() :
        Restore the original step size and momentum values.
    """

    def __init__(self, step_size, momentum):
        r"""
        Parameters
        ----------
        step_size : float
            The step size (learning rate) for the gradient descent.

        momentum : float
            The momentum factor to apply during updates.
        """

        self.step_size = step_size
        self.momentum  = momentum
        self.velocity  = 0

        self.temp_velocity = 0
        self._step_size    = step_size
        self._momentum     = momentum


    def apply_update(self, control, gradient, **kwargs):
        """
        Apply a gradient update to the control parameter.

        !!! note
            This is the steepest descent update: x_new = x_old - x_step.

        Parameters
        -------------------------------------------------------------------------------------
        control : array_like
            The current value of the parameter being optimized.

        gradient : array_like
            The gradient of the objective function with respect to the control parameter.

        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------------------------------------------------------------------------------------
        new_control, temp_velocity: tuple
            The new value of the control parameter after the update, and the current state step.
        """
        alpha = self._step_size
        beta  = self._momentum

        # apply update
        self.temp_velocity = beta*self.velocity - alpha*gradient
        new_control   = control + self.temp_velocity
        return new_control, self.temp_velocity

    def apply_smc_update(self, control, gradient, **kwargs):
        """
        Apply a gradient update to the control parameter.

        Parameters
        -------------------------------------------------------------------------------------
        control : array_like
            The current value of the parameter being optimized.

        gradient : array_like
            The gradient of the objective function with respect to the control parameter.

        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------------------------------------------------------------------------------------
        new_control: numpy.ndarray
            The new value of the control parameter after the update.
        """
        alpha = self._step_size

        # apply update
        new_control = (1-alpha) * control + alpha * gradient
        return new_control

    def apply_backtracking(self, shrink=0.5):
        """
        Apply backtracking by reducing step size and momentum temporarily.
        """
        self._step_size = shrink*self._step_size
        self._momentum  = shrink*self._momentum

    def restore_parameters(self):
        """
        Restore the original step size and momentum value.
        """
        self.velocity   = self.temp_velocity
        self._step_size = self.step_size
        self._momentum  = self.momentum

    def get_momentum_for_nesterov(self):
        """The momentum term, ``beta * velocity``, used for the Nesterov look-ahead."""
        return self.momentum * self.velocity

    def get_step_size(self):
        """Current step size."""
        return self._step_size


class Adam:
    """
    A class implementing the Adam optimizer for gradient-based optimization [`kingma2014`][].

    The Adam update equation for the control x using gradient g,
    iteration t, and small constants ε is given by:

        m_t = β1 * m_{t-1} + (1 - β1) * g   \n
        v_t = β2 * v_{t-1} + (1 - β2) * g^2 \n
        m_t_hat = m_t / (1 - β1^t)          \n
        v_t_hat = v_t / (1 - β2^t)          \n
        x_{t+1} = x_t - α * m_t_hat / (sqrt(v_t_hat) + ε)

    Attributes
    -------------------------------------------------------------------------------------
    step_size : float
        The initial step size provided during initialization.

    beta1 : float
        The exponential decay rate for the first moment estimates.

    beta2 : float
        The exponential decay rate for the second moment estimates.

    vel1 : 1-D array_like
        First moment estimate.

    vel2 : 1-D array_like
        Second moment estimate.

    eps : float
        Small constant to prevent division by zero.

    _step_size : float
        Private attribute for temporarily modifying step size.

    temp_vel1 : 1-D array_like
        Temporary first moment estimate.

    temp_vel2 : 1-D array_like
        Temporary Second moment estimate.

    Methods
    -------------------------------------------------------------------------------------
    apply_update(control, gradient, **kwargs):
        Apply an Adam update to the control parameter.

    apply_backtracking() :
        Apply backtracking by reducing step size temporarily.

    restore_parameters() :
        Restore the original step size.
    """

    def __init__(self, step_size, beta1=0.9, beta2=0.999):
        """
        A class implementing the Adam optimizer for gradient-based optimization.
        The Adam update equation for the control x using gradient g,
        iteration t, and small constants ε is given by:

            m_t = β1 * m_{t-1} + (1 - β1) * g   \n
            v_t = β2 * v_{t-1} + (1 - β2) * g^2 \n
            m_t_hat = m_t / (1 - β1^t)          \n
            v_t_hat = v_t / (1 - β2^t)          \n
            x_{t+1} = x_t - α * m_t_hat / (sqrt(v_t_hat) + ε)

        Parameters
        -------------------------------------------------------------------------------------
        step_size : float
            The step size (learning rate) for the optimization.

        beta1 : float, optional
            The exponential decay rate for the first moment estimates (default is 0.9).

        beta2 : float, optional
            The exponential decay rate for the second moment estimates (default is 0.999).

        """
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.vel1  = 0
        self.vel2  = 0
        self.eps   = 1e-7

        self._step_size = step_size
        self.temp_vel1 = 0
        self.temp_vel2 = 0

    def apply_update(self, control, gradient, **kwargs):
        """
        Apply a gradient update to the control parameter.

        !!! note
            This is the steepest descent update: x_new = x_old - x_step.

        Parameters
        -------------------------------------------------------------------------------------
        control : array_like
            The current value of the parameter being optimized.

        gradient : array_like
            The gradient of the objective function with respect to the control parameter.

        **kwargs : dict
            Additional keyword arguments, including 'iter' for the current iteration.

        Returns
        -------------------------------------------------------------------------------------
        new_control, temp_velocity: tuple
            The new value of the control parameter after the update, and the current state step.
        """
        iter  = kwargs['iter']
        alpha = self._step_size
        beta1 = self.beta1
        beta2 = self.beta2

        self.temp_vel1 = beta1*self.vel1 + (1-beta1)*gradient
        self.temp_vel2 = beta2*self.vel2 + (1-beta2)*gradient**2
        vel1_hat  = self.temp_vel1/(1-beta1**iter)
        vel2_hat  = self.temp_vel2/(1-beta2**iter)

        step = alpha*vel1_hat/(np.sqrt(vel2_hat)+self.eps)
        new_control = control - step  # steepest descent
        return new_control, step

    def apply_backtracking(self, shrink=0.5):
        """
        Apply backtracking by scaling the step size temporarily.
        """
        self._step_size = shrink*self._step_size

    def restore_parameters(self):
        """
        Restore the original step size.
        """
        self.vel1 = self.temp_vel1
        self.vel2 = self.temp_vel2
        self._step_size = self.step_size

    def get_step_size(self):
        """Current step size."""
        return self._step_size


class AdaMax(Adam):
    '''
    AdaMax optimizer [`kingma2014`][]
    '''
    def __init__(self, step_size, beta1=0.9, beta2=0.999):
        super().__init__(step_size, beta1, beta2)

    def apply_update(self, control, gradient, **kwargs):
        """An AdaMax step (Adam with the infinity norm on the second moment); returns ``(new_control, step)``."""
        iter  = kwargs['iter']
        alpha = self._step_size
        beta1 = self.beta1
        beta2 = self.beta2

        self.temp_vel1 = beta1*self.vel1 + (1-beta1)*gradient
        self.temp_vel2 = np.maximum(beta2*self.vel2, np.abs(gradient))

        step = alpha/(1-beta1**iter) * self.temp_vel1/self.temp_vel2
        new_control = control - step
        return new_control, step


class Steihaug:
    """
    A class implementing the Steihaug conjugate-gradient trust region optimizer. This code is based on the
    minfx optimisation library, https://gna.org/projects/minfx
    """

    def __init__(self, maxiter=1e6, epsilon=1e-8, delta_max=1e5, delta0=1.0):
        """
        Page 75 from 'Numerical Optimization' by Jorge Nocedal and Stephen J. Wright, 1999, 2nd ed.  The CG-Steihaug algorithm is:

        - epsilon > 0
        - p0 = 0, r0 = g, d0 = -r0
        - if ||r0|| < epsilon:
            - return p = p0
        - while 1:
            - if djT.B.dj <= 0:
                - Find tau such that p = pj + tau.dj minimises m(p) in (4.9) and satisfies ||p|| = delta
                - return p
            - aj = rjT.rj / djT.B.dj
            - pj+1 = pj + aj.dj
            - if ||pj+1|| >= delta:
                - Find tau such that p = pj + tau.dj satisfies ||p|| = delta
                - return p
            - rj+1 = rj + aj.B.dj
            - if ||rj+1|| < epsilon.||r0||:
                - return p = pj+1
            - bj+1 = rj+1T.rj+1 / rjT.rj
            - dj+1 = rj+1 + bj+1.dj

        Parameters
        -------------------------------------------------------------------------------------
        maxiter : float, optional
            Maximum number of iterations.

        epsilon : float, optional
            Tolerance for iterations.

        delta_max : float, optional
            Maximum thrust region size.

        delta0 : float, optional
            Initial thrust region size.
        """

        # Function arguments.
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.delta_max = delta_max
        self.delta0 = delta0
        self.delta = delta0

    def apply_update(self, xk, dfk, **kwargs):
        """
        Apply a Steihaug update to the control vector.

        Parameters
        -------------------------------------------------------------------------------------
        xk : array_like
            The current value of the parameter being optimized.

        dfk : array_like
            The gradient of the objective function with respect to the control parameter.

        **kwargs : dict
            Additional keyword arguments, including the hessian of the objective function
            with respect to the control parameter.

        Returns
        -------------------------------------------------------------------------------------
        new_control, step: tuple
            The new value of the control parameter after the update, and the current state step.
        """

        # Initial values at j = 0.
        d2fk = kwargs['hessian']
        pj = np.zeros(len(xk), np.float64)
        rj = dfk * 1.0
        dj = -dfk * 1.0
        B = d2fk * 1.0
        len_r0 = np.sqrt(np.dot(rj, rj))
        length_test = self.epsilon * len_r0

        log.debug("p0: " + repr(pj))
        log.debug("r0: " + repr(rj))
        log.debug("d0: " + repr(dj))

        if len_r0 < self.epsilon:
            log.debug("len rj < epsilon.")
            return xk, pj

        # Iterate over j.
        j = 0
        while True:
            # The curvature.
            curv = np.dot(dj, np.dot(B, dj))
            log.debug("Iteration j = " + repr(j))
            log.debug("Curv: " + repr(curv))

            # First test.
            if curv <= 0.0:
                tau = self.get_tau(rj, dj)
                log.debug("curv <= 0.0, therefore tau = " + repr(tau))
                pj_new = pj + tau * dj
                xk_new = xk + pj_new
                return xk_new, pj_new

            aj = np.dot(rj, rj) / curv
            pj_new = pj + aj * dj
            log.debug("aj: " + repr(aj))
            log.debug("pj+1: " + repr(pj_new))

            # Second test.
            if np.sqrt(np.dot(pj_new, pj_new)) >= self.delta:
                tau = self.get_tau(pj, dj)
                log.debug("sqrt(dot(self.pj_new, self.pj_new)) >= self.delta, therefore tau = "
                          + repr(tau))
                pj_new = pj + tau * dj
                xk_new = xk + pj_new
                return xk_new, pj_new

            rj_new = rj + aj * np.dot(B, dj)
            log.debug("rj+1: " + repr(rj_new))

            # Third test.
            if np.sqrt(np.dot(rj_new, rj_new)) < length_test:
                log.debug("sqrt(dot(self.rj_new, self.rj_new)) < length_test")
                xk_new = xk + pj_new
                return xk_new, pj_new

            bj_new = np.dot(rj_new, rj_new) / np.dot(rj, rj)
            dj_new = -rj_new + bj_new * dj
            log.debug("len rj+1: " + repr(np.sqrt(np.dot(rj_new, rj_new))))
            log.debug("epsilon.||r0||: " + repr(length_test))
            log.debug("bj+1: " + repr(bj_new))
            log.debug("dj+1: " + repr(dj_new))

            # Update j+1 to j.
            pj = pj_new * 1.0
            rj = rj_new * 1.0
            dj = dj_new * 1.0
            if j > self.maxiter:
                raise RuntimeError(f"Steihaug CG did not converge within {self.maxiter} iterations")
            j = j + 1

    def get_tau(self, pj, dj):
        """Function to find tau such that p = pj + tau.dj, and ||p|| = delta."""

        dot_pj_dj = np.dot(pj, dj)
        len_dj_sqrd = np.dot(dj, dj)

        # Positive root of ||pj + tau dj||^2 = delta^2. The whole numerator is
        # divided by ||dj||^2; dividing only the square root, as this once
        # did, put every boundary-hitting step at the wrong length.
        tau = (-dot_pj_dj + np.sqrt(
            dot_pj_dj ** 2 - len_dj_sqrd * (np.dot(pj, pj) - self.delta ** 2))) / len_dj_sqrd
        return tau

    def apply_backtracking(self, shrink=0.5):
        """
        Apply backtracking by scaling the trust radius temporarily.
        """
        self.delta = shrink * self.delta

    def restore_parameters(self):
        """
        Restore the original step size.
        """
        self.delta = self.delta0

    def get_step_size(self):
        """Current trust-region radius, which plays the role of the step size."""
        return self.delta
