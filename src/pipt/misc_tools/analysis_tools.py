"""
Collection of tools that can be used in update/analysis schemes.

Only put tools here that are so general that they can
be used by several update/analysis schemes. If some method is only applicable to the update scheme you are
implementing, leave it in that class.
"""

__all__ = [
    'parallel_upd',
    'calc_autocov',
    'calc_objectivefun'
]

# External imports
import os
import numpy as np          # Numerical tools
from scipy import linalg    # Linear algebra tools
from misc.system_tools.environ_var import OpenBlasSingleThread  # only single thread
import multiprocessing as mp  # parallel updates
import pickle
import logging
import warnings
from importlib import import_module  # To import packages

from scipy.spatial import cKDTree


def parallel_upd(list_state, prior_info, states_dict, X, local_mask_info, obs_data, pred_data, parallel, actnum=None,
                 field_dim=None, act_data_list=None, scale_data=None, num_states=1, emp_d_cov=False):
    """
    Script to initialize and control a parallel update of the ensemble state following [`emerick2016a`][].

    Parameters
    ----------
    list_state : list
        List of state names
    prior_info : dict
        INSERT DESCRIPTION
    states_dict : dict
        Dict. of state arrays
    X : ndarray
        INSERT DESCRIPTION
    local_mask_info : dict
        INSERT DESCRIPTION
    obs_data : ndarray
        Observed data
    pred_data : ndarray
        Predicted data
    parallel : int
        Number of parallel runs
    actnum : ndarray, optional
        Active cells
    field_dim : list, optional
        Number of grid cells in each direction
    act_data_list : list, optional
        List of active data names
    scale_data : ndarray, optional
        Scaling array for data
    num_states : int, optional
        Number of states
    emp_d_cov : bool
        INSERT DESCRIPTION

    Notes
    -----
    Since the localization matrix is to large for evaluation, we instead calculate it row for row.
    """
    if scale_data is None:
        scale_data = np.ones(obs_data.shape[0])

    # Generate a list over the grid coordinates
    if field_dim is not None:
        k_coord, j_coord, i_coord = np.meshgrid(range(field_dim[0]), range(
            field_dim[1]), range(field_dim[2]), indexing='ij')
        tot_g = np.array([k_coord, j_coord, i_coord])
        if actnum is not None:
            act_g = tot_g[:, actnum.reshape(field_dim)]
        else:
            act_g = tot_g[:, np.ones(tuple(field_dim), dtype=bool)]

    dat = [el for el in local_mask_info.keys()]
    # data coordinates to initialize search
    tot_completions = [tuple(el) for dat_mask in dat if isinstance(
        dat_mask, tuple) for el in local_mask_info[dat_mask]['position']]
    uniq_completions = [el for el in set(tot_completions)]
    tot_w_name = [dat_mask for dat_mask in dat if isinstance(
        dat_mask, tuple) for _ in local_mask_info[dat_mask]['position']]
    uniq_w_name = [tot_w_name[tot_completions.index(el)] for el in uniq_completions]
    # todo: limit to active datanan
    coord_search = cKDTree(data=uniq_completions)

    try:
        act_w_name = [el[0].split()[1] for el in uniq_w_name]

        tot_well_dict = {}
        for well in set(act_w_name):
            tot_well_dict[well] = [el for el in local_mask_info.keys() if isinstance(el, tuple) and
                                   el[0].split()[1] == well]
    except Exception:
        tot_well_dict = local_mask_info

    if len(scale_data.shape) == 1:
        diff = np.dot(np.expand_dims(scale_data**(-1), axis=1),
                      np.ones((1, pred_data.shape[1])))*(obs_data - pred_data)
    else:
        diff = linalg.solve(scale_data, (obs_data - pred_data))

    # initiallize the update
    upd = {}

    # Assume that we have three types of parameters. The full 3D fields, layers (2D fields), or scalar values. These are
    # handled individually.

    field_states = [state for state in list_state if states_dict[state].shape[0]
                    == act_g.shape[1]]  # field states
    layer_states = [state for state in list_state if 1 <
                    states_dict[state].shape[0] < act_g.shape[1]]  # layer states
    # scalar states
    scalar_states = [state for state in list_state if states_dict[state].shape[0] == 1]

    # We handle the field states first. These are the most time consuming, and requires parallelization.

    # since X must be passed to all processes I spit the state into equal portions, and let the row updates loop over
    # the different portions
    # coordinates for active parameters
    split_coord = np.array_split(act_g, parallel, axis=1)
    # Assuming that all parameters are spatial fields
    split_state = [{} for _ in range(parallel)]
    tmp_loc = {}  # intitallize for checking similar localization info
    # assume for now that everything is spatial, if not we require an extra loop or (if/else block)
    for state in field_states:
        # Augment the joint state variables (originally a dictionary) and the prior state variable
        aug_state = states_dict[state]
        # aug_prior_state = at.aug_state(self.prior_state, self.list_states)

        # Mean state and perturbation matrix
        mean_state = np.mean(aug_state, 1)
        if emp_d_cov:
            pert_state = (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                             np.ones((1, aug_state.shape[1]))))
        else:
            pert_state = (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                             np.ones((1, aug_state.shape[1])))) / (np.sqrt(aug_state.shape[1] - 1))

        tmp_state = np.array_split(pert_state, parallel)
        for i, elem in enumerate(tmp_state):
            split_state[i][state] = elem
        tmp_loc[state] = [el for el in local_mask_info if el[2] == state]
    # loc_info = [local_mask_info for _ in range(parallel)]
    # tot_X = [X for _ in range(parallel)]
    # tot_coord_seach = [coord_search for _ in range(parallel)] # might promt error if coord_search is to large
    # tot_uniq_name = [uniq_w_name for _ in range(parallel)]
    # tot_data_list = [act_data_list for _ in range(parallel)]
    # tot_well_dict_list = [tot_well_dict for _ in range(parallel)]
    non_similar = []
    for state in field_states[1:]:  # check localication
        non_shared = {k: ' ' for i, k in enumerate(
            tmp_loc[field_states[0]]) if local_mask_info[k] != local_mask_info[tmp_loc[state][i]]}
        non_similar.append(len(non_shared))

    if sum(non_similar) == 0:
        identical_loc = True
    else:
        identical_loc = False
    # Due to memory issues a pickle file is written containing all "meta" data required for the update
    with open('meta_analysis.p', 'wb') as file:
        pickle.dump({'local_mask_info': local_mask_info, 'diff': diff, 'X': X, 'coord_search': coord_search,
                     'unique_w_name': uniq_w_name, 'act_data_list': act_data_list, 'tot_well_dict': tot_well_dict,
                     'actnum': actnum, 'unique_completions': uniq_completions, 'identical_loc': identical_loc}, file)
    tot_file_name = ['meta_analysis.p' for _ in range(parallel)]
    # to_workers = zip(split_state, loc_info, diff, tot_X, split_coord, tot_coord_seach,tot_uniq_name, tot_data_list,
    #                  tot_well_dict_list)
    to_workers = zip(split_state, split_coord, tot_file_name)

    parallel = 1  # test
    #
    with OpenBlasSingleThread():
        if parallel > 1:
            with mp.get_context('spawn').Pool(parallel) as pool:
                s = pool.map(_calc_row_upd, to_workers)
        else:
            tmp_s = map(_calc_row_upd, to_workers)
            s = [el for el in tmp_s]

    for tmp_key in field_states:
        upd[tmp_key] = np.concatenate([el[tmp_key] for el in s], axis=0)

    ####################################################################################################################
    # Now handle the layer states

    for state in layer_states:
        # could add parallellizaton later
        aug_state = states_dict[state]
        mean_state = np.mean(aug_state, 1)
        if emp_d_cov:
            pert_state = {state: (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                     np.ones((1, aug_state.shape[1]))))}
        else:
            pert_state = {state: (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                     np.ones((1, aug_state.shape[1])))) / (np.sqrt(aug_state.shape[1] - 1))}
        # Layer
        # make a rule that requires the parameter name to end with the "_ + layer number". E.g. "multz_5"
        layer = int(state.split('_')[-1])
        l_act = np.full(field_dim, False)
        l_act[layer, :, :] = actnum.reshape(field_dim)[layer, :, :]
        act_g = tot_g[:, l_act]

        to_workers = zip([pert_state], [act_g], ['meta_analysis.p'])

        # with OpenBlasSingleThread():
        s = map(_calc_row_upd, to_workers)
        upd[state] = np.concatenate([el[state] for el in s], axis=0)

    ####################################################################################################################
    # Finally the scalar states
    for state in scalar_states:
        # could add parallellizaton later
        aug_state = states_dict[state]
        mean_state = np.mean(aug_state, 1)
        if emp_d_cov:
            pert_state = {state: (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                     np.ones((1, aug_state.shape[1]))))}
        else:
            pert_state = {state: (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                     np.ones((1, aug_state.shape[1])))) / (np.sqrt(aug_state.shape[1] - 1))}

        to_workers = zip([pert_state], [tot_g], ['meta_analysis.p'])

        # with OpenBlasSingleThread():
        s = map(_calc_row_upd, to_workers)

        upd[state] = np.concatenate([el[state] for el in s], axis=0)

    return upd


def _calc_row_upd(inp):
    """
    Calculate the updates.

    Parameters
    ----------
    inp : list
        List of [state, param_coordinates, metadata file name]
    """

    with open(inp[2], 'rb') as file:
        meta_data = pickle.load(file)
    states = [el for el in inp[0].keys()]
    Ne = inp[0][states[0]].shape[1]
    upd = {}
    for el in states:
        upd[el] = [np.zeros((1, Ne))]*(inp[0][el].shape[0])

    # Check and define regions for wells
    regions = _calc_region(meta_data['local_mask_info'], states,
                           meta_data['local_mask_info']['field'], meta_data['actnum'])
    max_r = {}
    for state in states:
        tmp_r = [meta_data['local_mask_info'][el]['range'][0] for el in meta_data['local_mask_info'].keys() if
                 isinstance(el, tuple) and state in el and
                 isinstance(meta_data['local_mask_info'][el]['range'][0], int)]
        if len(tmp_r):
            max_r[state] = max(tmp_r)
        else:
            max_r[state] = 0
    for i in range(inp[0][states[0]].shape[0]):
        for el in states:
            uniq_well = []
            if len(regions[el]):
                for reg in regions[el]:
                    if max_r[el] == 0:  # only use wells in the region, no taper
                        tmp_unique = []
                        for ind, w in enumerate(meta_data['unique_w_name']):
                            for comp in reg.T:
                                if meta_data['unique_completions'][ind][2] == comp[0] and \
                                        meta_data['unique_completions'][ind][1] == comp[1] and \
                                        meta_data['unique_completions'][ind][0] == comp[2]:
                                    tmp_unique.append(w)
                                    break
                        uniq_well.extend(tmp_unique)
                    else:  # only wells in the region, with taper
                        uniq_well.extend([w for w in set([meta_data['unique_w_name'][el] for el in
                                                          meta_data['coord_search'].query_ball_point(x=(inp[1][2, i], inp[1][1, i], inp[1][0, i]), r=max_r[el])])])
            else:
                uniq_well.extend([w for w in set([meta_data['unique_w_name'][el] for el in meta_data['coord_search'].query_ball_point(
                    x=(inp[1][2, i], inp[1][1, i], inp[1][0, i]), r=max_r[el])])])

            uniq_well = [(w[0], w[1], el) for w in set(uniq_well)]
            row_loc = np.zeros(meta_data['diff'].shape[0])
            for well in uniq_well:
                try:
                    tot_act_well = [elem for elem in meta_data['tot_well_dict']
                                    [well[0].split()[1]] if elem[2] == el]
                except Exception:
                    tot_act_well = [elem for elem in meta_data['tot_well_dict'][well]]
                # curr_completions = frozenset((inp[1][tot_act_well[0]]['position']))
                tot_act_data_types = set([el[0].split()[0] for el in tot_act_well])
                for data_typ in tot_act_data_types:
                    for el_well in tot_act_well:
                        if el_well[0].split()[0] == data_typ:
                            tmp_loc_info = el_well
                            break
                    curr_rho = _calc_loc(grid_pos=(inp[1][2, i], inp[1][1, i], inp[1][0, i]),
                                         loc_info=meta_data['local_mask_info'][tmp_loc_info], ne=Ne)
                    index = meta_data['act_data_list'][tmp_loc_info[0]]
                    row_loc[index] = curr_rho
                # for act_well in tot_act_well:
                #     # if len(curr_completions.difference(inp[1][act_well]['position'])) > 0:
                #     #     curr_completions = frozenset((inp[1][act_well]['position']))
                #     #     curr_rho = _calc_loc(grid_pos=(inp[4][2,i], inp[4][1,i], inp[4][0,i]), loc_info=inp[1][act_well],
                #     #                          ne=Ne)
                #     loc_index = inp[7][(act_well[0], act_well[1])]
                #     row_loc[loc_index] = curr_rho
            if 'identical_loc' in meta_data and meta_data['identical_loc']:
                for el_upd in states:
                    upd[el_upd][i] = np.dot(np.expand_dims(row_loc * np.dot(inp[0][el_upd][i, :], meta_data['X']), axis=0),
                                            meta_data['diff'])
                break
            else:
                upd[el][i] = np.dot(np.expand_dims(
                    row_loc*np.dot(inp[0][el][i, :], meta_data['X']), axis=0), meta_data['diff'])

    tot_upd = {}
    for el in states:
        tot_upd[el] = np.concatenate(upd[el], axis=0)

    return tot_upd


def _calc_region(loc_info, states, field_dim, actnum):
    """
    Calculate the region-boxes where data can be available for the state.

    Parameters
    ----------
    loc_info : dict
        Information for localization
    states : dict
        State variables
    field_dim : list
        Dimension of grid
    actnum : ndarray
        Active cells

    Returns
    -------
    regions : dict
        Region-box
    """
    regions = {}
    for state in states:
        tmp_reg = [loc_info[el]['range'] for el in loc_info.keys() if isinstance(el, tuple) and 'region' in loc_info[el]['taper_func']
                   and state in el]
        unique_reg = [el for el in set(map(tuple, tmp_reg))]
        regions[state] = []
        for reg in unique_reg:
            upd_reg = []
            for el in reg:
                # convert region boundaries (x0:x1) into list of integers [x0,x1]
                if ':' in el:
                    upd_reg.extend([int(l) for l in el.split(':')])
                else:
                    upd_reg.append(el)
            regions[state].append(_get_region(upd_reg, field_dim, actnum))

    return regions


def _get_region(reg, field_dim=None, actnum=None):
    """
    Calculate the coordinates of the region. Consider two formats.
    <ol>
        <li>k_min, k_max, j_min, j_max, i_min, i_max</li>
        <li>File (containing regions) regions</li>
    </ol>

    Parameters
    ----------
    reg :
    field_dim : list
        Dimension of grid
    actnum : ndarray
        Active cells

    Returns
    -------
    act_g : ndarray
    """

    # Get the files
    if isinstance(reg[0], str):
        flag_region = [int(el) for el in reg[1:]]
        with open(reg[0], 'r') as file:
            lines = file.readlines()
            # Extract all lines that start with a digit, and make a list of all digits
            tot_char = [el for l in lines if len(l.strip())
                        and l.strip()[0][0].isdigit() for el in l.split() if el[0].isdigit()]
        if field_dim is not None:
            # CHECK THIS AT SOME POINT!
            k_coord, j_coord, i_coord = np.meshgrid(range(field_dim[0]), range(
                field_dim[1]), range(field_dim[2]), indexing='ij')
            tot_g = np.array([k_coord, j_coord, i_coord])
            if actnum is not None:
                tot_f = np.zeros(field_dim).flatten()
                count = 0
                for l in tot_char:
                    if l.isdigit():
                        if int(l) in flag_region:
                            tot_f[count] = 1
                        count += 1
                    else:  # assume that we have input on the format num_cells*region_number
                        num_cell, tmp_region = l.split('*')
                        if int(tmp_region) in flag_region:
                            for i in range(int(num_cell)):
                                tot_f[count + i] = 1
                        count += int(num_cell)
                tot_f[~actnum] = 0
                act_g = tot_g[:, tot_f]
    else:
        # Get the domain
        if field_dim is not None:
            k_coord, j_coord, i_coord = np.meshgrid(range(field_dim[0]), range(
                field_dim[1]), range(field_dim[2]), indexing='ij')
            tot_g = np.array([k_coord, j_coord, i_coord])
            if actnum is not None:
                tot_f = np.zeros(field_dim, dtype=bool)
                tot_f[reg[4]:reg[5], reg[2]:reg[3], reg[0]:reg[1]] = actnum.reshape(
                    field_dim)[reg[4]:reg[5], reg[2]:reg[3], reg[0]:reg[1]]
                act_g = tot_g[:, tot_f]
            else:
                tot_f = np.zeros(field_dim, dtype=bool)
                tot_f[reg[4]:reg[5], reg[2]:reg[3], reg[0]:reg[1]] = np.ones(
                    field_dim, dtype=bool)[reg[4]:reg[5], reg[2]:reg[3], reg[0]:reg[1]]
                act_g = tot_g[:, tot_f]

    return act_g


def _calc_loc(grid_pos=[0, 0, 0], loc_info=None, ne=1):
    """
    _summary_

    Parameters
    ----------
    grid_pos : list, optional
     Grid coordinates. Defaults to [0,0,0].
    loc_info : dict, optional
        Localization inf. Defaults to None.
    ne : int, optional
        Number of ensemble members. Defaults to 1.

    Returns
    -------
    mask : ndarray
        Localization mask
    """
    # given the parameter type (to get the prior info) and the range to the data points we can calculate the
    # localization mask

    if loc_info['taper_func'] == 'region':
        mask = 1
    else:
        # TODO: Add 3D anisotropi
        loc_range = []
        for el in loc_info['position']:
            loc_range.append(_calc_dist(grid_pos, el))

        dist = min(loc_range)
        if loc_info['taper_func'] == 'fb':
            # assume that FB localization is utilized. Here vi can add all different localization functions
            if dist < loc_info['range'][0]:
                tmp = 1 - 1 * \
                    (1.5 * np.abs(dist) / loc_info['range']
                     [0] - .5 * (dist / loc_info['range'][0]) ** 3)
            else:
                tmp = 0

            mask = (ne * tmp ** 2) / ((tmp ** 2) * (ne + 1) + 1 ** 2)

    return mask


def _calc_dist(x1, x2):
    """
    Calculate distance between two points

    Parameters
    ----------
    x1, x2: ndarray
        Coordinates

    Returns
    -------
    dist : ndarray
        (Euclidean) distance between `x1` and `x2`

    """
    if len(x1) == 1:
        return np.sqrt((x1-x2)**2)
    elif len(x1) == 2:
        return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
    elif len(x1) == 3:
        return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2)


def calc_autocov(pert):
    """
    Calculate sample auto-covariance matrix.

    Parameters
    ----------
    pert : ndarray
        Perturbation matrix (matrix of variables perturbed with their mean)

    Returns
    -------
    cov_auto : ndarray
        Sample auto-covariance matrix
    """
    # TODO: Implement sqrt-covariance matrices

    # No of samples
    ne = pert.shape[1]

    # Standard sample auto-covariance calculation
    cov_auto = (1 / (ne - 1)) * np.dot(pert, pert.T)

    # Return the auto-covariance matrix
    return cov_auto

def calc_objectivefun(pert_obs, pred_data, Cd):
    """
    Calculate the objective function.

    Parameters
    ----------
    pert_obs : array-like
        NdxNe array containing perturbed observations.

    pred_data : array-like
        NdxNe array containing ensemble of predictions.

    Cd : array-like
        NdxNd array containing data covariance, or Ndx1 array containing data variance.

    Returns
    -------
    data_misfit : array-like
        Nex1 array containing objective function values.
    """
    #ne = pred_data.shape[1]
    ne = pert_obs.shape[1]
    r = (pred_data[:, :ne] - pert_obs)  # Only use ne members (gies code has ne+1 predicted data)
    # The per-member misfit is the diagonal of r.T @ (Cd^-1 r). Summing the
    # columns gives the same numbers without forming the (ne, ne) product.
    if len(Cd.shape) == 1:
        precision = Cd**(-1)
        data_misfit = np.sum(r * (r*precision[:, None]), axis=0)
    else:
        data_misfit = np.sum(r * linalg.solve(Cd, r), axis=0)

    return data_misfit


def save_assimilation_result(ind_save, **kwargs):
    """
    Save the requested variables for one assimilation iteration.

    The PIPT counterpart to ``popt.misc_tools.optim_tools.save_optimize_results``,
    which writes ``optimize_result_{i}.npz``.

    Parameters
    ----------
    ind_save : int
        Iteration index. ``0`` is the prior.
    **kwargs : dict
        Variables that will be saved to npz file

    Notes
    -----
    Use kwargs here because the input will be a dictionary with names equal the variable names to store, and when this
    is passed to np.savez (kwargs) the variable will be stored with their original name.
    """
    # Save input variables
    folder = kwargs.pop('savefolder')
    os.makedirs(folder, exist_ok=True)
    try:
        np.savez(f'{folder}/assimilation_result_{ind_save}', **kwargs)
    except Exception: # if npz save fails dump to a pickle file
        with open(f'{folder}/assimilation_result_{ind_save}.p', 'wb') as file:
            pickle.dump(kwargs, file)


def save_analysisdebug(ind_save, **kwargs):
    """Deprecated alias for :func:`save_assimilation_result`.

    The files are not a debugging aid -- they are the per-iteration record of
    a run -- so both the function and what it writes were renamed.
    """
    warnings.warn(
        "save_analysisdebug is deprecated; use save_assimilation_result. "
        "Note that it now writes 'assimilation_result_{i}.npz' rather than "
        "'debug_analysis_step_{i}.npz'.",
        DeprecationWarning,
        stacklevel=2,
    )
    return save_assimilation_result(ind_save, **kwargs)


def get_list_data_types(obs_data, assim_index):
    """
    Extract the list of all and active data types

    Parameters
    ----------
    obs_data : dict
        Observed data
    assim_index : int
        Current assimilation index

    Returns
    -------
    l_all : list
        List of all data types
    l_act : list
        List of the data types that are active (that are not `None`)
    """
    # List the primary indices
    if isinstance(assim_index[0], list):  # If True, then we have subset list
        if isinstance(assim_index[1][0], list):  # Check if prim. ind. is a list
            l_prim = [int(x) for x in assim_index[1][0]]
        else:
            l_prim = [int(assim_index[1][0])]
    else:  # Only prim. assim. ind.
        if isinstance(assim_index[1], list):  # Check if prim. ind. is a list
            l_prim = [int(x) for x in assim_index[1]]
        else:
            l_prim = [int(assim_index[1])]

    # List the data types.
    l_all = list(obs_data[l_prim[0]].keys())

    # Extract the data types that are active at current assimilation step
    l_act = []
    for ix in l_prim:
        for data_typ in l_all:
            if obs_data[ix][data_typ] is not None:
                l_act.extend([data_typ])

    # Return the list
    return l_all, l_act


def gen_covdata(datavar, assim_index, list_data):
    """
    Generate the data covariance matrix at current assimilation step. Note here that the data covariance may be a
    diagonal matrix with only variance entries, or an empirical covariance matrix, or both if in combination. For
    diagonal data covariance we only store vector of variance values.

    Parameters
    ----------
    datavar : list
        List of dictionaries containing variance for the observed data. The structure of this list is the same as for
        `obs_data`
    assim_index : int
        Current assimilation index
    list_data : list
        List of the data types

    Returns
    -------
    cd : ndarray
        Data auto-covariance matrix

    Notes
    -----
    For empirical covariance generation, the datavar entry must be a 2D array, arranged as a standard ensemble matrix (N
    x Ns, where Ns is the number of samples).
    """
    # TODO: Change if sub-assim. indices are implemented
    # TODO: Use something other that numpy hstack for this augmentation!

    # Make sure assim_index is list
    if isinstance(assim_index[1], list):  # Check if prim. ind. is a list
        l_prim = [int(x) for x in assim_index[1]]
    else:
        l_prim = [int(assim_index[1])]

    # Init. a logical variable to check if it is the first time in the loop below that we extract variance data.
    # Need this because we stack the remaining variance horizontally, and it is possible that we have "None"
    # input in the first instances of the loop (hence we cannot always say that
    # self.datavar[l_prim[0]][list_data[0]] will be the first variance we want to extract)
    first_time = True

    # Initialize augmented array
    # Loop over all primary indices
    for ix in range(len(l_prim)):
        # Loop over data types and augment the data variance
        for i in range(len(list_data)):
            if datavar[l_prim[ix]][list_data[i]] is not None:
                # If there is an observed data here, augment it
                if first_time:  # Init. var output
                    # Switch off the first time logical variable
                    first_time = False

                    # Calc. var.
                    var = datavar[l_prim[ix]][list_data[i]]

                    # If var is 2D then it is either full covariance or realizations to generate a sample cov.
                    # If matrix is square assume it is full covariance, note this can go wrong!
                    if var.ndim == 2:
                        if var.shape[0] == var.shape[1]:  # full cov
                            c_var = var
                        else:
                            c_var = calc_autocov(var)
                    # else we make a diagonal matrix
                    else:  # diagonal, only store vector
                        c_var = var

                else:  # Stack var output
                    # Calc. var.
                    var = datavar[l_prim[ix]][list_data[i]]

                    # If var is 2D then we generate a sample cov., else we make a diagonal matrix
                    if var.ndim == 2:  # empirical
                        if var.shape[0] == var.shape[1]:  # full cov
                            c_var_temp = var
                        else:
                            c_var_temp = calc_autocov(var)
                        c_var = linalg.block_diag(c_var, c_var_temp)
                    else:  # diagonal, only store vector
                        c_var_temp = var
                        c_var = np.append(c_var, c_var_temp)

    # Generate the covariance matrix
    cd = c_var

    # Return data covariance matrix
    return cd


def store_ensemble_sim_information(saveinfo, member):
    """
    Here, we can either run a unique python script or do some other post-processing routines. The function should
    not return anything, but provide a method for storing revevant information.
    Input the current member for easy storage
    """

    for el in saveinfo:
        if '.py' in el:  # This is a unique python file
            sim_info_func = import_module(el[:-3])  # remove .py ending
            # Note: the function must be named main, and we pass the full current instance of the object pluss the
            # current member.
            sim_info_func.main(member)


def aug_obs_pred_data(obs_data, pred_data, assim_index, list_data):
    """
    Augment the observed and predicted data to an array at an assimilation step. The observed data will be an augemented
    vector and the predicted data will be an ensemble matrix.

    Parameters
    ----------
    obs_data : list
        List of dictionaries containing observed data
    pred_data : list
        List of dictionaries where each entry of the list is the forward simulation results at an assimilation step. The
        dictionary has keys equal to the data type (given in `OBSNAME`).

    Returns
    -------
    obs : ndarray
        Augmented vector of observed data
    pred : ndarray
        Ensemble matrix of predicted data
    """
    # TODO: Change if sub-assim. ind. are implemented.
    # TODO: Use something other that numpy hstack and vstack for these augmentations!

    # Make sure assim_index is a list
    if isinstance(assim_index[1], list):  # Check if prim. ind. is a list
        l_prim = [int(x) for x in assim_index[1]]
    else:
        l_prim = [int(assim_index[1])]

    # make this more efficient

    tot_pred = tuple(pred_data[el][dat] for el in l_prim if pred_data[el]
                     is not None for dat in list_data if obs_data[el][dat] is not None)

    if len(tot_pred):  # if this is done during the initiallization tot_pred contains nothing
        pred = np.concatenate(tot_pred)
    else:
        pred = None
    obs = np.concatenate(tuple(
        obs_data[el][dat] for el in l_prim for dat in list_data if obs_data[el][dat] is not None))

    # Init. a logical variable to check if it is the first time in the loop below that we extract obs/pred data.
    # Need this because we stack the remaining data horizontally/vertically, and it is possible that we have "None"
    # input in the first instances of the loop (hence we cannot always say that
    # self.obs_data[l_prim[0]][list_data[0]] and self.pred_data[l_prim[0]][list_data[0]] will be the
    # first data we want to extract)
    # first_time = True
    #
    # #initialize obs and pred
    # obs = None
    # pred = None
    #
    # # Init the augmented arrays.
    # # Loop over all primary indices
    # for ix in range(len(l_prim)):
    #     # Loop over obs_data/pred_data keys
    #     for i in range(len(list_data)):
    #         # If there is an observed data here, augment obs and pred
    #         if obs_data[l_prim[ix]][list_data[i]] is not None:  # No obs/pred data
    #             if first_time:  # Init. the outputs obs and pred
    #                 # Switch off the first time logical variable
    #                 first_time = False
    #
    #                 # Observed data:
    #                 obs = obs_data[l_prim[ix]][list_data[i]]
    #
    #                 # Predicted data
    #                 pred = pred_data[l_prim[ix]][list_data[i]]
    #
    #             else:  # Stack the obs and pred outputs
    #                 # Observed data:
    #                 obs = np.hstack((obs, obs_data[l_prim[ix]][list_data[i]]))
    #
    #                 # Predicted data
    #                 pred = np.vstack((pred, pred_data[l_prim[ix]][list_data[i]]))
    #
    # # Return augmented arrays
    return obs, pred


def aug_state(state, list_state, cell_index=None):
    """
    Augment the state variables to an array.

    Parameters
    ----------
    state : dict
        Dictionary of initial ensemble of (joint) state variables (static parameters and dynamic variables) to be
        assimilated.
    list_state : list
        Fixed list of keys in state dict.
    cell_index : list of vector indexes to be extracted

    Returns
    -------
    aug : ndarray
        Ensemble matrix of augmented state variables
    """
    # TODO: In some rare cases, it may not be desirable to update every state variable at each assimilation step.
    # Change code to only augment states to be updated at the specific assimilation step
    # TODO: Use something other that numpy vstack for this augmentation!

    if cell_index is not None:
        # Start with ensemble of first state variable
        aug = state[list_state[0]][cell_index]

        # Loop over the next states (if exists)
        for i in range(1, len(list_state)):
            aug = np.vstack((aug, state[list_state[i]][cell_index]))

        # Return the augmented array

    else:
        # Start with ensemble of first state variable
        aug = state[list_state[0]]

        # Loop over the next states (if exists)
        for i in range(1, len(list_state)):
            aug = np.vstack((aug, state[list_state[i]]))

        # Return the augmented array
    return aug


def calc_scaling(enX, idX, prior_info):
    """
    Form the scaling to be used in svd related algoritms. Scaling consist of standard deviation for each `STATICVAR`
    It is important that this is formed in the same manner as the augmentet state vector is formed. Hence, with the same
    list of states.

    Parameters
    ----------
    enX : np.ndarray
        State ensemble matrix, shape ``(nx, ne)``; only its row count per
        variable is used.
    idX : dict
        Row range ``(start, stop)`` of each state variable in ``enX``, in the
        order the state was stacked.
    prior_info : dict
        Nested dictionary containing prior information

    Returns
    -------
    scaling : numpy array
        scaling
    """

    scaling = []
    for elem in idX.keys():
        # more than single value. This is for multiple layers. Assume all values are active
        if len(prior_info[elem]['variance']) > 1:
            ny = prior_info[elem]['ny']
            nx = prior_info[elem]['nx']
            scaling.append(np.tile(np.sqrt(prior_info[elem]['variance']), ny*nx))
        else:
            i = idX[elem][0]
            j = idX[elem][1]
            ones = np.ones(enX[i:j].shape[0])
            scaling.append(np.sqrt(prior_info[elem]['variance']) * ones)

    return np.concatenate(scaling)


def update_state(aug_state, state, list_state, cell_index=None):
    """
    Extract the separate state variables from an augmented state array. It is assumed that the augmented state
    array is made in `aug_state`, hence this is the reverse method of `aug_state`.

    Parameters
    ----------
    aug_state : ndarray
        Augmented array of UPDATED state variables
    state : dict
        Dict. of state variables NOT updated.
    list_state : list
        List of state keys that have been updated
    cell_index : list
        List of indexes that gives the where the aug state should be placed

    Returns
    -------
    state : dict
        Dict. of UPDATED state variables
    """
    if cell_index is None:
        # Loop over all entries in list_state and extract a matrix with same number of rows as the key in state
        # determines from aug and replace the values in state[key].
        # Init. a variable to keep track of which row in 'aug' we start from in each loop
        aug_row = 0
        for _, key in enumerate(list_state):
            # Find no. rows in state[lkey] to determine how many rows from aug to extract
            no_rows = state[key].shape[0]

            # Extract the rows from aug and update 'state[key]'
            state[key] = aug_state[aug_row:aug_row + no_rows, :]

            # Update tracking variable for row in 'aug'
            aug_row += no_rows

    else:
        aug_row = 0
        for _, key in enumerate(list_state):
            # Find no. rows in state[lkey] to determine how many rows from aug to extract
            no_rows = len(cell_index)

            # Extract the rows from aug and update 'state[key]'
            state[key][cell_index, :] = aug_state[aug_row:aug_row + no_rows, :]

            # Update tracking variable for row in 'aug'
            aug_row += no_rows
    return state


def limits(state, prior_info):
    """
    Check if any state variables overshoots the limits given by the prior info. If so, modify these values

    Parameters
    ----------
    state : dict
        Dictionary containing the states
    prior_info : dict
        Dictionary containing prior information for all the states.

    Returns
    -------
    state : dict
        Valid state
    """
    for var in state.keys():
        if 'limits' in prior_info[var]:
            state[var][state[var] < prior_info[var]['limits'][0]] = prior_info[var]['limits'][0]
            state[var][state[var] > prior_info[var]['limits'][1]] = prior_info[var]['limits'][1]
    return state


def truncSVD(matrix, r=None, energy=None, full_matrices=False):
    '''
    Perform truncated SVD on input matrix.

    Parameters
    ----------
    matrix : ndarray, shape (m, n)
        Input matrix to perform SVD on.

    r : int, optional
        Rank to truncate the SVD to. If None, energy must be specified.

    energy : float, optional
        Fraction of the singular-value sum to retain, given either as a fraction
        in (0, 1] or as a percentage in (1, 100]. The smallest rank whose
        retained fraction reaches this value is used, so the requested amount is
        met rather than approached from below. Note this accumulates the
        singular values themselves, not their squares -- it is a fraction of the
        nuclear norm, not of the Frobenius energy. If None, r must be specified.

    full_matrices : bool, optional
        Whether to compute full or reduced SVD. Default is False.

    Returns
    -------
    U : ndarray, shape (m, r)
        Left singular vectors.

    S : ndarray, shape (r,)
        Singular values.

    VT : ndarray, shape (r, n)
        Right singular vectors transposed.
    '''
    # Perform SVD on input matrix
    U, S, VT = np.linalg.svd(matrix, full_matrices=full_matrices)

    # If not specified rank, energy must be given
    if r is None:
        if energy is None:
            raise ValueError("Either rank 'r' or 'energy' must be specified for truncSVD.")

        # Accept a percentage (1, 100] as well as a fraction (0, 1]. The bound is
        # exclusive so that energy=1 keeps everything rather than meaning 1%.
        fraction = energy/100 if energy > 1 else energy

        total = np.sum(S)
        if total == 0:
            # No spectrum to apportion; nothing is more representative than
            # anything else, so keep it all rather than dividing by zero.
            r = len(S)
        else:
            # searchsorted gives the first index at which the cumulative
            # fraction REACHES `fraction`; that index must be kept, hence +1.
            # Clamped here rather than below so that energy=1 does not trip the
            # "specified rank" warning on a rounding error in the last entry.
            r = min(int(np.searchsorted(np.cumsum(S)/total, fraction)) + 1, len(S))

    if r == 0:
        r = 1  # Ensure at least one singular value is retained
    if r > len(S):
        warnings.warn("Specified rank exceeds the number of singular values; using all of them.", stacklevel=2)
        r = len(S)

    return U[:,:r], S[:r], VT[:r,:]

def get_outlier_index(
    pred,
    data,
    data_var=None,
    tresh=4.0
):
    """
    Identify outlier ensemble members based on a normalized data-mismatch score.

    For each ensemble member j, the mismatch is:

        h_j = sum_i ((Y_ij - d_i) / sigma_i)^2

    where sigma_i is the ensemble standard deviation (or provided variance) for observable i.
    Members whose score deviates more than `tresh` standard deviations from the mean are flagged as outliers.

    Parameters
    ----------
    pred : array_like, shape (nd, ne)
        Predicted data ensemble, one column per member.
    data : array_like, shape (nd,)
        Observed data, in the same row order.
    data_var : array_like or None, optional
        Data variance, ``(nd,)`` or ``(nd, ne)`` for an empirical ensemble. If not provided, the ensemble
        variance of the predicted data is used.
    tresh : float, optional
        Outlier threshold in numbers of standard deviations. Default is 4.

    Returns
    -------
    outlier_indices : np.ndarray
        Indices of outlier ensemble members.
    members : np.ndarray
        Array of ensemble member indices, with outliers replaced by randomly selected non-outlier members.
    """
    Y = np.asarray(pred, dtype=float)  # (nd, ne)
    d = np.asarray(data, dtype=float).reshape(-1, 1)  # (nd, 1)

    # Determine variance for normalization
    if data_var is not None:
        var = np.asarray(data_var, dtype=float)
        if var.ndim == 1:
            var = var[:, np.newaxis]
    else:
        var = np.var(Y, axis=1, ddof=1)[:, np.newaxis]

    # Compute normalized data-mismatch score for each ensemble member
    mismatch = np.sum(((Y - d) / np.sqrt(var)) ** 2, axis=0)  # (ne,)

    # Identify outliers using the sigma rule
    mean_mismatch = np.mean(mismatch)
    std_mismatch = np.std(mismatch)
    outlier_mask = np.abs(mismatch - mean_mismatch) > tresh * std_mismatch
    outlier_indices = np.where(outlier_mask)[0]
    non_outlier_members = np.where(~outlier_mask)[0]

    if len(outlier_indices) > 0:
        logging.getLogger(__name__).info(f" Identified outliers: {outlier_indices}")

    return outlier_indices, non_outlier_members
