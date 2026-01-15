"""
ES, and Iterative ES updates with hybrid update matrix calculated from multi-fidelity runs.
"""

import numpy as np
from scipy.linalg import solve
from pipt.misc_tools import analysis_tools as at

class hybrid_update:
    '''
    Class for hybrid update schemes as described in: Fossum, K., Mannseth, T., & Stordal, A. S. (2020). Assessment of
    multilevel ensemble-based data assimilation for reservoir history matching. Computational Geosciences, 24(1),
    217–239. https://doi.org/10.1007/s10596-019-09911-x

    Note that the scheme is slightly modified to be inline with the standard (I)ES approximate update scheme. This
    enables the scheme to efficiently be coupled with multiple updating strategies via class MixIn
    '''

    def scale(self, data, scaling):
        """
        Scale the data perturbations by the data error standard deviation.

        Args:
            data (np.ndarray): data perturbations
            scaling (np.ndarray): data error standard deviation

        Returns:
            np.ndarray: scaled data perturbations
        """

        if len(scaling.shape) == 1:
            return (scaling ** (-1))[:, None] * data
        else:
            return solve(scaling, data)
        
    def update(self, enX, enY, enE, **kwargs):
        '''
        Perform the hybrid update.

        Parameters:
        ----------
            enX : list of np.ndarray 
                List of state ensemble matrices for each level (nx, ne)
            
            enY : list of np.ndarray
                List of predicted data ensemble matrices for each level (nd, ne)
            
            enE : list of np.ndarray
                List of ensemble of perturbed observations for each level (nd, ne)
        '''
        # Loop over levels to calculate the update step
        X3 = []
        enXcentered = []
        for l in range(self.tot_level):
            
            # Get Perturbed state ensemble at level l
            if ('emp_cov' in self.keys_da) and (self.keys_da['emp_cov'] == 'yes'):
                enXcentered.append(self.scale(enX[l] - np.mean(enX[l], 1)[:,None], self.state_scaling))
            else:
                enXcentered.append(self.scale(np.dot(enX[l], self.proj[l]), self.state_scaling))

            # Calculate truncated SVD of predicted data ensemble at level l 
            enYcentered = self.scale(np.dot(enY[l], self.proj[l]), self.scale_data[l])
            Ud, Sd, VTd = at.truncSVD(enYcentered, energy=self.trunc_energy)

            X2 = solve(((self.lam + 1)*np.eye(len(Sd)) + np.diag(Sd**2)), Ud.T)
            X3.append(np.dot(np.dot(VTd.T, np.diag(Sd)), X2))
                
        # Calculate each row of self.step individually to avoid memory issues.
        self.step = [np.empty(enXcentered[l].shape) for l in range(self.tot_level)]
        step_size = min(1000, int(self.state_scaling.shape[0]/2)) # do maximum 1000 rows at a time.

        # Generate row batches
        nrows = self.state_scaling.shape[0]
        row_step = [np.arange(s, min(s + step_size, nrows)) for s in range(0, nrows, step_size)]

        # Loop over rows
        for row in row_step:
            ml_weights = self.multilevel['ml_weights']
            kg = sum([ml_weights[l]*np.dot(enXcentered[l][row, :], X3[l]) for l in range(self.tot_level)])

            # Loop over levels
            for l in range(self.tot_level):
                enRes = self.scale(enE[l] - enY[l], self.scale_data[l])
                self.step[l][row, :] = np.dot(self.state_scaling[row, None] * kg, enRes)


