"""Gradient acceleration."""
import numpy as np


class GradientAscent:
    r"""
    A class for performing gradient ascent optimization with momentum and backtracking.
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
            The step size (learning rate) for the gradient ascent.

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
            This is the steepest decent update: x_new = x_old - x_step.

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

    def apply_backtracking(self):
        """
        Apply backtracking by reducing step size and momentum temporarily.
        """
        self._step_size = 0.5*self._step_size
        self._momentum  = 0.5*self._momentum
    
    def restore_parameters(self):
        """
        Restore the original step size and momentum value.
        """
        self.velocity   = self.temp_velocity
        self._step_size = self.step_size
        self._momentum  = self.momentum
    
    def get_momentum_for_nesterov(self):
        return self.momentum * self.velocity

    def get_step_size(self):
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
            This is the steepest decent update: x_new = x_old - x_step.

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
        new_control = control - step  # steepest decent
        return new_control, step

    def apply_backtracking(self):
        """
        Apply backtracking by reducing step size temporarily.
        """
        self._step_size = 0.5*self._step_size
    
    def restore_parameters(self):
        """
        Restore the original step size.
        """
        self.vel1 = self.temp_vel1
        self.vel2 = self.temp_vel2
        self._step_size = self.step_size

    def get_step_size(self):
        return self._step_size


class AdaMax(Adam):
    '''
    AdaMax optimizer [`kingma2014`][]
    '''
    def __init__(self, step_size, beta1=0.9, beta2=0.999):
        super().__init__(step_size, beta1, beta2)
    
    def apply_update(self, control, gradient, **kwargs):
        iter  = kwargs['iter'] 
        alpha = self._step_size
        beta1 = self.beta1
        beta2 = self.beta2

        self.temp_vel1 = beta1*self.vel1 + (1-beta1)*gradient
        self.temp_vel2 = np.maximum(beta2*self.vel2, np.abs(gradient))
        
        step = alpha/(1-beta1**iter) * self.temp_vel1/self.temp_vel2
        new_control = control - step 
        return new_control, step
    
