"""
ES, and Iterative ES updates with hybrid update matrix calculated from multi-fidelity runs.
"""

import numpy as np
from scipy.linalg import solve
from pipt.misc_tools import analysis_tools as at
import pipt.misc_tools.extract_tools as extract
from pipt.update_schemes.analysis.base import AnalysisBase, AnalysisResult

class hybrid_update(AnalysisBase):
    '''
    Class for hybrid update schemes as described in: Fossum, K., Mannseth, T., & Stordal, A. S. (2020). Assessment of
    multilevel ensemble-based data assimilation for reservoir history matching. Computational Geosciences, 24(1),
    217–239. https://doi.org/10.1007/s10596-019-09911-x

    Note that the scheme is slightly modified to be inline with the standard (I)ES approximate update scheme. This
    is what lets it be bound as an analysis like ``approx_update`` and friends, despite working on *lists* of
    per-level matrices rather than single ones -- see ``esmda_hybrid.COMPATIBLE_ANALYSES``.
    '''

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
        # esmda_hybrid computes its own proj/scale_data (one entry per
        # fidelity level, where other flavours have a single matrix). Reading
        # them off the scheme picks those up automatically -- that is what
        # the scheme's own_or_ensemble properties are for.
        scheme = self.scheme
        proj = scheme.proj
        scale_data = scheme.scale_data
        state_scaling = scheme.state_scaling

        # Loop over levels to calculate the update step
        X3 = []
        enXcentered = []
        for l in range(scheme.tot_level):

            # Get Perturbed state ensemble at level l
            if extract.is_enabled(scheme.keys_da.get('emp_cov', False)):
                enXcentered.append(self.solve(state_scaling, enX[l] - np.mean(enX[l], 1)[:,None]))
            else:
                enXcentered.append(self.solve(state_scaling, np.dot(enX[l], proj[l])))

            # Calculate truncated SVD of predicted data ensemble at level l
            enYcentered = self.solve(scale_data[l], np.dot(enY[l], proj[l]))
            Ud, Sd, VTd = at.truncSVD(enYcentered, energy=scheme.trunc_energy)

            X2 = solve(((scheme.lam + 1)*np.eye(len(Sd)) + np.diag(Sd**2)), Ud.T)
            X3.append(np.dot(np.dot(VTd.T, np.diag(Sd)), X2))

        # Calculate each row of step individually to avoid memory issues.
        step = [np.empty(enXcentered[l].shape) for l in range(scheme.tot_level)]
        # Generate row batches: at most 1000 rows at a time, and at least one,
        # so a single-row state does not produce an empty range.
        nrows = state_scaling.shape[0]
        step_size = max(1, min(1000, nrows // 2))
        row_step = [np.arange(s, min(s + step_size, nrows)) for s in range(0, nrows, step_size)]

        # Loop over rows
        for row in row_step:
            ml_weights = scheme.multilevel['ml_weights']
            kg = sum([ml_weights[l]*np.dot(enXcentered[l][row, :], X3[l]) for l in range(scheme.tot_level)])

            # Loop over levels
            for l in range(scheme.tot_level):
                enRes = self.solve(scale_data[l], enE[l] - enY[l])
                step[l][row, :] = np.dot(state_scaling[row, None] * kg, enRes)

        return AnalysisResult(step=step)
