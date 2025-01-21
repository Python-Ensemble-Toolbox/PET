"""
ES-MDA type schemes
"""

# External imports
import scipy.linalg as scilinalg
from copy import deepcopy
import numpy as np
from geostat.decomp import Cholesky

# Internal imports
from pipt.loop.ensemble import Ensemble
import pipt.misc_tools.analysis_tools as at

# import update schemes
from pipt.update_schemes.update_methods_ns.approx_update import approx_update
from pipt.update_schemes.update_methods_ns.full_update import full_update
from pipt.update_schemes.update_methods_ns.subspace_update import subspace_update


class esmdaMixIn(Ensemble):
    """
    This is the implementation of the ES-MDA algorithm given in [`emerick2013a`][].
    This algorithm have been implemented mostly to
    illustrate how a algorithm using the Mda loop can be implemented.
    """

    def __init__(self, keys_da, keys_en, sim):
        """
        The class is initialized by passing the keywords and simulator object upwards in the hierarchy.

        Parameters
        ----------
        keys_da['mda'] : list
            - tot_assim_steps: total number of iterations in MDA, e.g., 3
            - inflation_param: covariance inflation factors, e.g., [2, 4, 4]

        keys_en : dict

        sim : callable
        """
        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_da, keys_en, sim)

        self.prev_data_misfit = None

        if self.restart is False:
            self.prior_state = deepcopy(self.state)
            self.list_states = list(self.state.keys())
            # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.check_assimindex_simultaneous()
            self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(
                self.obs_data, self.assim_index)

            # Extract no. assimilation steps from MDA keyword in DATAASSIM part of init. file and set this equal to
            # the number of iterations pluss one. Need one additional because the iter=0 is the prior run.
            self.max_iter = len(self._ext_assim_steps())+1
            self.iteration = 0
            self.lam = 0  # set LM lamda to zero as we are doing one full update.
            if 'energy' in self.keys_da:
                # initial energy (Remember to extract this)
                self.trunc_energy = self.keys_da['energy']
                if self.trunc_energy > 1:  # ensure that it is given as percentage
                    self.trunc_energy /= 100.
            else:
                self.trunc_energy = 0.98
            # Get the perturbed observations and observation scaling
            self._ext_obs()
            self.real_obs_data_conv = deepcopy(self.real_obs_data)
            # Get state scaling and svd of scaled prior
            self._ext_state()
            self.current_state = deepcopy(self.state)
        # Extract the inflation parameter from MDA keyword
        self.alpha = self._ext_inflation_param()

        self.prev_data_misfit = None

    def calc_analysis(self):
        r"""
        Analysis step of ES-MDA. The analysis algorithm is similar to EnKF analysis, only difference is that the data
        covariance matrix is inflated with an inflation parameter alpha. The update is done as an iterative smoother
        where all data is assimilated at once.

        Notes
        -----
        ES-MDA is an iterative ensemble smoother with a predefined number of iterations, where the updates is done with
        the EnKF update equations but where the data covariance matrix have been inflated:

        $$ \begin{align}
        d_{obs} &= d_{true} + \sqrt{\alpha}C_d^{1/2}Z \\
        m &= m_{prior} + C_{md}(C_g + \alpha C_d)^{-1}(g(m) - d_{obs})
        \end{align} $$

        where $d_{true}$ is the true observed data, $\alpha$ is the inflation factor, $C_d$ is the data covariance
        matrix, $Z$ is a standard normal random variable, $C_{md}$ and $C_{g}$ are sample covariance matrices,
        $m$ is the model parameter, and $g(\)$ is the predicted data. Note that $\alpha$ can have a different
        value in each assimilation step and must fulfill:

        $$ \sum_{i=1}^{N_a} \frac{1}{\alpha} = 1 $$

        where $N_a$ being the total number of assimilation steps.
        """
        # Get assimilation order as a list
        # reformat predicted data
        _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                     self.list_datatypes)

        init_en = Cholesky()  # Initialize GeoStat class for generating realizations
        if self.iteration == 1:  # first iteration
            data_misfit = at.calc_objectivefun(
                self.real_obs_data_conv, self.aug_pred_data, self.cov_data)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.prior_data_misfit_std = np.std(data_misfit)
            self.data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            self.logger.info(
                f'Prior run complete with data misfit: {self.prior_data_misfit:0.1f}.')
            self.data_random_state = deepcopy(np.random.get_state())
            self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector,
                                                                   self.alpha[self.iteration-1] *
                                                                   self.cov_data, self.ne,
                                                                   return_chol=True)
            self.E = np.dot(self.real_obs_data, self.proj)
        else:
            self.data_random_state = deepcopy(np.random.get_state())
            self.obs_data_vector, _ = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                           self.list_datatypes)
            self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector,
                                                                   self.alpha[self.iteration -
                                                                              1] * self.cov_data,
                                                                   self.ne,
                                                                   return_chol=True)
            self.E = np.dot(self.real_obs_data, self.proj)

        if 'localanalysis' in self.keys_da:
            self.local_analysis_update()
        else:
            if len(self.scale_data.shape) == 1:
                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
            else:
                self.pert_preddata = scilinalg.solve(
                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

            aug_state = at.aug_state(self.current_state, self.list_states)

            self.update()
            if hasattr(self, 'step'):
                aug_state_upd = aug_state + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                aug_prior_state = at.aug_state(self.prior_state, self.list_states)
                aug_state_upd = np.dot(aug_prior_state, (np.eye(
                    self.ne) + self.W / np.sqrt(self.ne - 1)))

            # Extract updated state variables from aug_update
            self.state = at.update_state(aug_state_upd, self.state, self.list_states)
            self.state = at.limits(self.state, self.prior_info)

    def check_convergence(self):
        """
        Check if LM-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter.

        Returns
        -------
        bool
            Logic variable telling if algorithm has converged
        dict
            Dict. with keys corresponding to conv. criteria, with logical variable telling which of them that has been
            met
        """

        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                          self.list_datatypes)

        data_misfit = at.calc_objectivefun(
            self.real_obs_data_conv, pred_data, self.cov_data)
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        if self.data_misfit < self.prev_data_misfit:
            self.logger.info(
                f'MDA iteration number {self.iteration}! Objective function reduced from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        else:
            self.logger.info(
                f'MDA iteration number {self.iteration}! Objective function increased from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        # Return conv = False, why_stop var.
        self.current_state = deepcopy(self.state)
        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        return False, True, why_stop

    def _ext_inflation_param(self):
        r"""
        Extract the data covariance inflation parameter from the MDA keyword in DATAASSIM part. Also, we check that
        the criterion:

        $$ \sum_{i=1}^{N_a} \frac{1}{\alpha} = 1 $$

        is fulfilled for the inflation factor, alpha. If the keyword for inflation parameter -- INFLATION_PARAM -- is
        not provided, we set the default $\alpha_i = N_a$, where $N_a$ is the tot. no. of MDA assimilation steps (the
        criterion is fulfilled with this value).

        Returns
        -------
        alpha: list
            Data covariance inflation factor
        """
        # Make sure MDA is a list
        if not isinstance(self.keys_da['mda'][0], list):
            mda_opts = [self.keys_da['mda']]
        else:
            mda_opts = self.keys_da['mda']

        # Check if INFLATION_PARAM has been provided, and if so, extract the value(s). If not, we set alpha to the
        # default value equal to the tot. no. assim. steps
        if 'inflation_param' in list(zip(*mda_opts))[0]:
            # Extract value
            alpha_tmp = [item[1] for item in mda_opts if item[0] == 'inflation_param'][0]

            # If one value is given, we copy it to all assim. steps. If multiple values are given, we check the
            # number of parameters corresponds to tot. no. assim. steps
            if not isinstance(alpha_tmp, list):  # Single input
                alpha = [alpha_tmp] * len(self._ext_assim_steps())  # Copy value

            else:
                assert len(alpha_tmp) == len(self._ext_assim_steps()), 'Number of parameters given in INFLATION_PARAM in MDA does ' \
                    'not match the total number of assimilation steps given by ' \
                    'TOT_ASSIM_STEPS in same keyword!'

                # Inflation parameters for each assimilation step given directly
                alpha = alpha_tmp

        else:  # Give alpha by default value
            alpha = [len(self._ext_assim_steps())] * len(self._ext_assim_steps())

        # Check if alpha fulfills the criterion to machine precision
        assert 1 - np.finfo(float).eps <= sum([(1 / x) for x in alpha]) <= 1 + np.finfo(float).eps, \
            'The sum of the inverse of the inflation parameters given in INFLATION_PARAM does not add up to 1!'

        # Return inflation parameter
        return alpha

    def _ext_assim_steps(self):
        """
        Extract list of assimilation steps to perform in MDA loop from the MDA keyword (mandatory for
        MDA class) in DATAASSIM part. (This method is similar to Iterative._ext_max_iter)

        Parameters
        ----------
        keys_da : dict
            all keywords from DATAASSIM part
        mda : info
            for MDA methods

        Returns
        -------
        int
            Total number of MDA assimilation steps

        Changelog
        ---------
        - ST 7/6-16
        - ST 1/3-17: Changed to output list of assim. steps instead of just tot. assim. steps
        """
        # Make sure MDA is a list
        if not isinstance(self.keys_da['mda'][0], list):
            mda_opts = [self.keys_da['mda']]
        else:
            mda_opts = self.keys_da['mda']

        # Check if 'max_iter' has been given; if not, give error (mandatory in ITERATION)
        assert 'tot_assim_steps' in list(
            zip(*mda_opts))[0], 'TOT_ASSIM_STEPS has not been given in MDA!'

        # Extract max. iter
        tot_no_assim = int([item[1]
                           for item in mda_opts if item[0] == 'tot_assim_steps'][0])

        # Make a list of assim. steps
        assim_steps = list(range(tot_no_assim))

        # If it is a restart run, we remove simulations already done
        if self.restart is True:
            # List simulations we already have done. Do this by checking pred_data.
            # OBS: Minus 1 here do to the aborted simulation is also not None.
            # TODO: Relying on loop_ind may not be the best strategy (?)
            sim_done = list(range(self.loop_ind))

            # Update list of assim. steps by removing simulations we have done
            assim_steps = [ind for ind in assim_steps if ind not in sim_done]

        # Return list assim. steps
        return assim_steps


class esmda_approx(esmdaMixIn, approx_update):
    pass


class esmda_full(esmdaMixIn, full_update):
    pass


class esmda_subspace(esmdaMixIn, subspace_update):
    pass


class esmda_geo(esmda_approx):
    """
    This is the implementation of the ES-MDA-GEO algorithm from [1]. The main analysis step in this algorithm is the
    same as the standard ES-MDA algorithm (implemented in the `es_mda` class). The difference between this and the
    standard algorithm is the calculation of the inflation factor. Also see [`rafiee2017`][].
    """

    def __init__(self, keys_da):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_da)

        # Within
        self.alpha = [None] * self.tot_assim

    def _calc_inflation_factor(self, pert_preddata, cov_data, energy=99):
        """
        We calculate the inflation factor, follow the procedure laid out in Algorithm 1 in [1].

        Parameters
        ----------
        pert_preddata : ndarray
            Predicted data (fwd. run) ensemble matrix perturbed with its mean
        cov_data : ndarray
            Data covariance matrix
        energy : float, optional
            Percentage of energy kept in (T)SVD decompostion of 'sensitivity' matrix (default is 99%)

        Returns
        -------
        alpha : float
            Inflation factor
        beta : float
            Geometric factor
        """
        # Need the square-root of the data covariance matrix
        if np.count_nonzero(cov_data - np.diagonal(cov_data)) == 0:
            l = np.sqrt(cov_data)  # only variance (diagonal) term
        else:
            # Cholesky decomposition
            l = scilinalg.cholesky(cov_data)  # cov. matrix has off-diag. terms

        # Calculate the 'sensitivity' matrix:
        sens = (1 / np.sqrt(self.ne - 1)) * np.dot(l, pert_preddata)

        # Perform SVD on sensitivtiy matrix
        _, s_d, _ = np.linalg.svd(sens, full_matrices=False)

        # If no. measurements is more than ne - 1, we only keep ne - 1 sing. val.
        if sens.shape[0] >= self.ne:
            s_d = s_d[:-1].copy()

        # If energy is less than 100 we truncate the SVD matrices
        if energy < 100:
            ti = (np.cumsum(s_d) / sum(s_d)) * 100 <= energy
            s_d = s_d[ti].copy()

        # Calc average singular value
        avg_s_d = s_d.mean()

        # The inflation factor is chosen as the maximum of the average singular value (squared) and max. no. of
        # iterations
        alpha = np.max((avg_s_d ** 2, self.tot_assim))

        # We calculate the geometric (reduction) factor (called 'common ratio' in the article). The formula is given
        # as (1 - beta**-n) / (1 - beta**-1) = alpha (it is actually incorrect in the article, and should be as
        # written here), with n=tot. assim. steps. Rewritten:
        #
        # (1-alpha)*beta**n + alpha*beta**(n-1) - 1 = 0
        #
        # This is of course a nasty polynomial root problem, but we use Numpy.roots, extract the real
        # root less than 1, and hope for the best :p
        root_coeff = np.zeros(self.tot_assim + 1)
        root_coeff[0] = 1 - alpha  # first coeff. in polynomial
        root_coeff[1] = alpha  # sec. coeff in polynomial
        root_coeff[-1] = -1
        roots = np.roots(root_coeff)

        # Most likely the first root will be 1, and the second one will be the one we want. Due to numerical
        # imprecision, the first root will not be exactly one, so we us Numpy.min to get the second root.
        beta = np.min([x.real for x in roots if x.imag == 0 and x.real < 1])

        # Return inflation and geometric factor
        return alpha, beta
