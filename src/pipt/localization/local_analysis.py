"""Local-analysis localization strategy. Not functional at present; see the CHANGELOG's Known issues."""
import pipt.misc_tools.analysis_tools as at
import numpy as np
from typing import Union
from scipy.spatial import distance
from pipt.localization.common import (
    LocalizationBase,
    LocalizationConfigBuilder,
)

__all__ = ["LocalAnalysisLocalization", "_calc_loc", "_calc_distance"]


class LocalAnalysisLocalization(LocalizationBase):
    """Local-analysis strategy carrying mode-specific localization metadata."""

    name = "localanalysis"

    def __init__(
            self,
            info: Union[dict, list],
            data_indices: Union[list, None] = None,
            data_types: Union[list, None] = None,
            parameters: Union[list, None] = None,
            ensemble_size: Union[int, None] = None,
        ):
        """
        Initialize the LocalAnalysisLocalization instance.

        Parameters
        ----------
        info : dict or list
            Localization configuration information.
        data_indices : list
            Indices of the data to be assimilated.
        data_types : list
            Types of the data to be assimilated.
        parameters : list
            List of free parameters for the assimilation.
        ensemble_size : int
            Size of the ensemble used in the assimilation.
        """
        config = LocalizationConfigBuilder(info)
        loc_info = config.build(
            data_index=data_indices,
            data_types=data_types,
            parameters=parameters,
            ne=ensemble_size,
        )
        super().__init__(loc_info)


def _calc_loc(max_dist, distance, prior_info, loc_type, ne):
    """Compute local-analysis weights for distance-based localization."""
    variance = prior_info["variance"][0]
    mask = np.zeros(len(distance))

    if loc_type == "fb":
        for i in range(len(distance)):
            if distance[i] < max_dist:
                tmp = variance - variance * (
                    1.5 * np.abs(distance[i]) / max_dist - 0.5 * (distance[i] / max_dist) ** 3
                )
            else:
                tmp = 0
            mask[i] = (ne * tmp ** 2) / ((tmp ** 2) * (ne + 1) + variance ** 2)

    elif loc_type == "gc":
        for count, dist_value in enumerate(np.abs(distance)):
            if dist_value <= max_dist:
                tmp = (
                    -(1.0 / 4.0) * (dist_value / max_dist) ** 5
                    + (1.0 / 2.0) * (dist_value / max_dist) ** 4
                    + (5.0 / 8.0) * (dist_value / max_dist) ** 3
                    - (5.0 / 3.0) * (dist_value / max_dist) ** 2
                    + 1
                )
            elif dist_value <= 2 * max_dist:
                tmp = (
                    (1.0 / 12.0) * (dist_value / max_dist) ** 5
                    - (1.0 / 2.0) * (dist_value / max_dist) ** 4
                    + (5.0 / 8.0) * (dist_value / max_dist) ** 3
                    + (5.0 / 3.0) * (dist_value / max_dist) ** 2
                    - 5.0 * (dist_value / max_dist)
                    + 4.0
                    - (2.0 / 3.0) * (max_dist / dist_value)
                )
            else:
                tmp = 0.0
            mask[count] = tmp

    return mask[np.newaxis, :]

def _calc_distance(data_pos, index_unique, current_data_list, assim_index, obs_data, pred_data, param_pos):
    """
    Calculate the distance between data and parameters.

    Parameters
    ----------
    data_pos : dict
        Dictionary containing the position of the data.

    index_unique : bool
        Boolean that determines if the position is unique.

    current_data_list : list
        List containing the names of the data that should be evaluated.

    assim_index : int
        The index of the data to be evaluated.

    obs_data : list of dict
        List of dictionaries containing the data.

    pred_data : list of dict
        List of dictionaries containing the predictions.

    param_pos : list of tuple
        List of tuples representing the position of the parameters.

    Returns
    -------
        - dist: list of euclidean distance between the data/parameter pair.
    """
    # distance to data if distance based localization
    if index_unique is False:
        dist = []
        for dat in current_data_list:
            for indx in assim_index[1]:
                indx_data_pos = data_pos[dat][indx]
                if obs_data[indx] is not None and obs_data[indx][dat] is not None:
                    # add shortest distance
                    dist.append(min(distance.cdist(indx_data_pos, param_pos).flatten()))
    else:
        dist = []
        for data in current_data_list:
            elem_data_pos = data_pos[data]
            obs, _ = at.aug_obs_pred_data(obs_data, pred_data, assim_index, [data])
            dist.extend(
                len(obs)*[min(distance.cdist(elem_data_pos, param_pos).flatten())])

    return dist
