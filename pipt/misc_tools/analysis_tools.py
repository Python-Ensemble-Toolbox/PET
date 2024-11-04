"""
Collection of tools that can be used in update/analysis schemes.

Only put tools here that are so general that they can
be used by several update/analysis schemes. If some method is only applicable to the update scheme you are
implementing, leave it in that class.
"""

# External imports
import numpy as np          # Numerical tools
from scipy import linalg    # Linear algebra tools
from misc.system_tools.environ_var import OpenBlasSingleThread  # only single thread
import multiprocessing as mp  # parallel updates
import time
import pickle
from importlib import import_module  # To import packages

from scipy.spatial import cKDTree


def parallel_upd(list_state, prior_info, states_dict, X, local_mask_info, obs_data, pred_data, parallel, actnum=None,
                 field_dim=None, act_data_list=None, scale_data=None, num_states=1, emp_d_cov=False):
    """
    Script to initialize and control a parallel update of the ensemble state following [1].

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

    References
    ----------
    [1] Emerick, Alexandre A. 2016. “Analysis of the Performance of Ensemble-Based Assimilation of Production and
    Seismic Data.” Journal of Petroleum Science and Engineering 139. Elsevier: 219-39. doi:10.1016/j.petrol.2016.01.029
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
    tot_completions = [tuple(el) for dat_mask in dat if type(
        dat_mask) == tuple for el in local_mask_info[dat_mask]['position']]
    uniq_completions = [el for el in set(tot_completions)]
    tot_w_name = [dat_mask for dat_mask in dat if type(
        dat_mask) == tuple for _ in local_mask_info[dat_mask]['position']]
    uniq_w_name = [tot_w_name[tot_completions.index(el)] for el in uniq_completions]
    # todo: limit to active datanan
    coord_search = cKDTree(data=uniq_completions)

    try:
        act_w_name = [el[0].split()[1] for el in uniq_w_name]

        tot_well_dict = {}
        for well in set(act_w_name):
            tot_well_dict[well] = [el for el in local_mask_info.keys() if type(el) == tuple and
                                   el[0].split()[1] == well]
    except:
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
                 type(el) == tuple and state in el and type(meta_data['local_mask_info'][el]['range'][0]) == int]
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
                except:
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
        tmp_reg = [loc_info[el]['range'] for el in loc_info.keys() if type(el) == tuple and 'region' in loc_info[el]['taper_func']
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
    if type(reg[0]) == str:
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
    ne = pred_data.shape[1]
    r = (pred_data - pert_obs)
    if len(Cd.shape) == 1:
        precission = Cd**(-1)
        data_misfit = np.diag(r.T.dot(r*precission[:, None]))
    else:
        data_misfit = np.diag(r.T.dot(linalg.solve(Cd, r)))

    return data_misfit


def calc_crosscov(pert1, pert2):
    """
    Calculate sample cross-covariance matrix.

    Parameters
    ----------
    pert1, pert2: ndarray
        Perturbation matrices (matrix of variables perturbed with their mean).

    Returns
    -------
    cov_cross : ndarray
        Sample cross-covariance matrix
    """
    # TODO: Implement sqrt-covariance matrices

    # No of samples
    ne = pert1.shape[1]

    # Standard calc. of sample cross-covariance
    cov_cross = (1 / (ne - 1)) * np.dot(pert1, pert2.T)

    # Return the cross-covariance matrix
    return cov_cross


def update_datavar(cov_data, datavar, assim_index, list_data):
    """
    Extract the separate variance from an augmented vector. It is assumed that the augmented variance
    is made gen_covdata, hence this is the reverse method of gen_covdata.

    Parameters
    ----------
    cov_data : array-like
        Augmented vector of variance.

    datavar : dict
        Dictionary of separate variances.

    assim_index : list
        Assimilation order as a list.

    list_data : list
        List of data keys.

    Returns
    -------
    datavar : dict
        Updated dictionary of separate variances."""

    # Loop over all entries in list_state and extract a vector with same number of elements as the key in datavar
    # determines from aug and replace the values in datavar[key].

    # Make sure assim_index is list
    if isinstance(assim_index[1], list):  # Check if prim. ind. is a list
        l_prim = [int(x) for x in assim_index[1]]
    else:
        l_prim = [int(assim_index[1])]

    # Extract the diagonal if cov_data is a matrix
    if len(cov_data.shape) == 2:
        cov_data = np.diag(cov_data)

    # Initialize a variable to keep track of which row in 'cov_data' we start from in each loop
    aug_row = 0
    # Loop over all primary indices
    for ix in range(len(l_prim)):
        # Loop over data types and augment the data variance
        for i in range(len(list_data)):
            if datavar[l_prim[ix]][list_data[i]] is not None:

                # If there is an observed data here, update it
                no_rows = datavar[l_prim[ix]][list_data[i]].shape[0]

                # Extract the rows from aug and update 'state[key]'
                datavar[l_prim[ix]][list_data[i]] = cov_data[aug_row:aug_row + no_rows]

                # Update tracking variable for row in 'aug'
                aug_row += no_rows

    # Return
    return datavar


def save_analysisdebug(ind_save, **kwargs):
    """
    Save variables in analysis step for debugging purpose

    Parameters
    ----------
    ind_save : int
        Index of analysis step
    **kwargs : dict
        Variables that will be saved to npz file

    Notes
    -----
    Use kwargs here because the input will be a dictionary with names equal the variable names to store, and when this
    is passed to np.savez (kwargs) the variable will be stored with their original name.
    """
    # Save input variables
    try:
        np.savez('debug_analysis_step_{0}'.format(str(ind_save)), **kwargs)
    except: # if npz save fails dump to a pickle file
        with open(f'debug_analysis_step_{ind_save}.p', 'wb') as file:
            pickle.dump(kwargs, file)


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


def screen_data(cov_data, pred_data, obs_data_vector, keys_da, iteration):
    """
    INSERT DESCRIPTION

    Parameters
    ----------
    cov_data : ndarray
        Data covariance matrix
    pred_data : ndarray
        Predicted data
    obs_data_vector : 
        Observed data (1D array)
    keys_da : dict
        Dictionary with every input in `DATAASSIM`
    iteration : int
        Current iteration

    Returns
    -------
    cov_data : ndarray
        Updated data covariance matrix
    """

    if ('restart' in keys_da and keys_da['restart'] == 'yes') or (iteration != 0):
        with open('cov_data.p', 'rb') as f:
            cov_data = pickle.load(f)
    else:
        emp_cov = False
        if cov_data.ndim == 2:  # assume emp_cov
            emp_cov = True
            var = np.var(cov_data, ddof=1, axis=1)
            cov_data = cov_data - cov_data.mean(1)[:, np.newaxis]
        num_data = pred_data.shape[0]
        for i in range(num_data):
            v = 0
            if obs_data_vector[i] < np.min(pred_data[i, :]):
                v = np.abs(obs_data_vector[i] - np.min(pred_data[i, :]))
            elif obs_data_vector[i] > np.max(pred_data[i, :]):
                v = np.abs(obs_data_vector[i] - np.max(pred_data[i, :]))
            if not emp_cov:
                cov_data[i] = np.max((cov_data[i], v ** 2))
            else:
                v = np.max((v**2 / var[i], 1))
                cov_data[i, :] *= np.sqrt(v)
        with open('cov_data.p', 'wb') as f:
            pickle.dump(cov_data, f)

    return cov_data


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


def extract_tot_empirical_cov(data_var, assim_index, list_data, ne):
    """
    Extract realizations of noise from data_var (if imported), or generate realizations if only variance is specified
    (assume uncorrelated)

    Parameters
    ----------
    data_var : list
        List of dictionaries containing the varianse as read from the input
    assim_index : int
        Index of the assimilation
    list_data : list
        List of data types
    ne : int
        Ensemble size

    Returns
    -------
    E : ndarray
        Sorted (according to assim_index and list_data) matrix of data realization noise.
    """

    if isinstance(assim_index[1], list):  # Check if prim. ind. is a list
        l_prim = [int(x) for x in assim_index[1]]
    else:
        l_prim = [int(assim_index[1])]

    tmp_E = []
    for el in l_prim:
        tmp_tmp_E = {}
        for dat in list_data:
            if data_var[el][dat] is not None:
                if len(data_var[el][dat].shape) == 1:
                    tmp_tmp_E[dat] = np.sqrt(
                        data_var[el][dat][:, np.newaxis])*np.random.randn(data_var[el][dat].shape[0], ne)
                else:
                    if data_var[el][dat].shape[0] == data_var[el][dat].shape[1]:
                        tmp_tmp_E[dat] = np.dot(linalg.cholesky(
                            data_var[el][dat]), np.random.randn(data_var[el][dat].shape[1], ne))
                    else:
                        tmp_tmp_E[dat] = data_var[el][dat]
        tmp_E.append(tmp_tmp_E)
    E = np.concatenate(tuple(tmp_E[i][dat] for i, el in enumerate(
        l_prim) for dat in list_data if data_var[el][dat] is not None))

    return E


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


def calc_kalmangain(cov_cross, cov_auto, cov_data, opt=None):
    r"""
    Calculate the Kalman gain

    Parameters
    ----------
    cov_cross : ndarray
        Cross-covariance matrix between state and predicted data
    cov_auto : ndarray
        Auto-covariance matrix of predicted data
    cov_data : ndarray
        Variance on observed data (diagonal matrix)
    opt : str
        Which method should we use to calculate Kalman gain
        <ul>
            <li>'lu': LU decomposition (default)</li>
            <li>'chol': Cholesky decomposition</li>
        </ul>

    Returns
    -------
    kalman_gain : ndarray
        Kalman gain

    Notes
    -----
    In the following Kalman gain is $K$, cross-covariance is $C_{mg}$, predicted data auto-covariance is $C_{g}$,
    and data covariance is $C_{d}$.

    With `'lu'` option, we solve the transposed linear system:
    $$
        K^T = (C_{g} + C_{d})^{-T}C_{mg}^T
    $$

    With `'chol'` option we use Cholesky on auto-covariance matrix,
    $$
       L L^T = (C_{g} + C_{d})^T
    $$
    and solve linear system with the square-root matrix from Cholesky:
    $$
        L^T Y = C_{mg}^T\\
        LK = Y
    $$
    """
    if opt is None:
        calc_opt = 'lu'

    # Add data and predicted data auto-covariance matrices
    if len(cov_data.shape) == 1:
        cov_data = np.diag(cov_data)
    c_auto = cov_auto + cov_data

    if calc_opt == 'lu':
        kg = linalg.solve(c_auto.T, cov_cross.T)
        kalman_gain = kg.T

    elif calc_opt == 'chol':
        # Cholesky decomp (upper triangular matrix)
        u = linalg.cho_factor(c_auto.T, check_finite=False)

        # Solve linear system with cholesky square-root
        kalman_gain = linalg.cho_solve(u, cov_cross.T, check_finite=False)

    # Return Kalman gain
    return kalman_gain


def calc_subspace_kalmangain(cov_cross, data_pert, cov_data, energy):
    """
    Compute the Kalman gain in a efficient subspace determined by how much energy (i.e. percentage of singluar values)
    to retain. For more info regarding the implementation, see Chapter 14 in [1].

    Parameters
    cov_cross : ndarray
        Cross-covariance matrix between state and predicted data
    data_pert : ndarray
            Predicted data - mean of predicted data
    cov_data : ndarray
        Variance on observed data (diagonal matrix)

    Returns
    -------
    k_g : ndarray
        Subspace Kalman gain

    References
    ----------
    [1] G. Evensen (2009). Data Assimilation: The Ensemble Kalman Filter, Springer.
    """
    # No. ensemble members
    ne = data_pert.shape[1]

    # Perform SVD on pred. data perturbations
    u_d, s_d, v_d = np.linalg.svd(np.sqrt(1 / (ne - 1)) * data_pert, full_matrices=False)

    # If no. measurements is more than ne - 1, we only keep ne - 1 sing. val.
    if data_pert.shape[0] >= ne:
        u_d, s_d, v_d = u_d[:, :-1].copy(), s_d[:-1].copy(), v_d[:-1, :].copy()

    # If energy is less than 100 we truncate the SVD matrices
    if energy < 100:
        ti = (np.cumsum(s_d) / sum(s_d)) * 100 <= energy
        u_d, s_d, v_d = u_d[:, ti].copy(), s_d[ti].copy(), v_d[ti, :].copy()

    # Calculate x_0 and its eigenvalue decomp.
    if len(cov_data.shape) == 1:
        x_0 = np.dot(np.diag(s_d[:]**(-1)), np.dot(u_d[:, :].T, np.expand_dims(cov_data, axis=1)*np.dot(u_d[:, :],
                                                                                                        np.diag(s_d[:]**(-1)).T)))
    else:
        x_0 = np.dot(np.diag(s_d[:] ** (-1)), np.dot(u_d[:, :].T, np.dot(cov_data, np.dot(u_d[:, :],
                                                                                          np.diag(s_d[:] ** (-1)).T))))
    s, u = np.linalg.eig(x_0)

    # Calculate x_1
    x_1 = np.dot(u_d[:, :], np.dot(np.diag(s_d[:]**(-1)).T, u))

    # Calculate Kalman gain based on the subspace matrices we made above
    k_g = np.dot(cov_cross, np.dot(x_1, linalg.solve(
        (np.eye(s.shape[0]) + np.diag(s)), x_1.T)))

    # Return subspace Kalman gain
    return k_g


def compute_x(pert_preddata, cov_data, keys_da, alfa=None):
    """
    INSERT DESCRIPTION

    Parameters
    ----------
    pert_preddata : ndarray
        Perturbed predicted data
    cov_data : ndarray
        Data covariance matrix
    keys_da : dict
        Dictionary with every input in `DATAASSIM`
    alfa : None, optional
        INSERT DESCRIPTION

    Returns : 
    X : ndarray
        INSERT DESCRIPTION
    """
    X = []
    if 'kalmangain' in keys_da and keys_da['kalmangain'][0] == 'subspace':

        # TSVD energy
        energy = keys_da['kalmangain'][1]

        # No. ensemble members
        ne = pert_preddata.shape[1]

        # Calculate x_0 and its eigenvalue decomp.
        if len(cov_data.shape) == 1:
            scale = np.expand_dims(np.sqrt(cov_data), axis=1)
        else:
            scale = np.expand_dims(np.sqrt(np.diag(cov_data)), axis=1)

        # Perform SVD on pred. data perturbations
        u_d, s_d, v_d = np.linalg.svd(pert_preddata/scale, full_matrices=False)

        # If no. measurements is more than ne - 1, we only keep ne - 1 sing. val.
        if pert_preddata.shape[0] >= ne:
            u_d, s_d, v_d = u_d[:, :-1].copy(), s_d[:-1].copy(), v_d[:-1, :].copy()

        # If energy is less than 100 we truncate the SVD matrices
        if energy < 100:
            ti = (np.cumsum(s_d) / sum(s_d)) * 100 <= energy
            u_d, s_d, v_d = u_d[:, ti].copy(), s_d[ti].copy(), v_d[ti, :].copy()

        # Calculate x_0 and its eigenvalue decomp.
        if len(cov_data.shape) == 1:
            x_0 = np.dot(np.diag(s_d[:] ** (-1)),
                         np.dot(u_d[:, :].T, np.expand_dims(cov_data, axis=1) * np.dot(u_d[:, :],
                                                                                       np.diag(s_d[:] ** (-1)).T)))
        else:
            x_0 = np.dot(np.diag(s_d[:] ** (-1)), np.dot(u_d[:, :].T, np.dot(cov_data, np.dot(u_d[:, :],
                                                                                              np.diag(s_d[:] ** (-1)).T))))
        s, u = np.linalg.eig(x_0)

        # Calculate x_1
        x_1 = np.dot(u_d[:, :], np.dot(np.diag(s_d[:] ** (-1)).T, u))/scale

        # Calculate X based on the subspace matrices we made above
        X = np.dot(np.dot(pert_preddata.T, x_1), linalg.solve(
            (np.eye(s.shape[0]) + np.diag(s)), x_1.T))

    else:
        if len(cov_data.shape) == 1:
            X = linalg.solve(np.dot(pert_preddata, pert_preddata.T) +
                             np.diag(cov_data), pert_preddata)
        else:
            X = linalg.solve(np.dot(pert_preddata, pert_preddata.T) +
                             cov_data, pert_preddata)
        X = X.T

    return X


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


def calc_scaling(state, list_state, prior_info):
    """
    Form the scaling to be used in svd related algoritms. Scaling consist of standard deviation for each `STATICVAR`
    It is important that this is formed in the same manner as the augmentet state vector is formed. Hence, with the same
    list of states.

    Parameters
    ----------
    state : dict
        Dictionary containing the state
    list_state : list
        List of states for augmenting
    prior_info : dict
        Nested dictionary containing prior information

    Returns
    -------
    scaling : numpy array
        scaling
    """

    scaling = []
    for elem in list_state:
        # more than single value. This is for multiple layers. Assume all values are active
        if len(prior_info[elem]['variance']) > 1:
            scaling.append(np.concatenate(tuple(np.sqrt(prior_info[elem]['variance'][z]) *
                                                np.ones(
                                                    prior_info[elem]['ny']*prior_info[elem]['nx'])
                                                for z in range(prior_info[elem]['nz']))))
        else:
            scaling.append(tuple(np.sqrt(prior_info[elem]['variance']) *
                                 np.ones(state[elem].shape[0])))

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


def resample_state(aug_state, state, list_state, new_en_size):
    """
    Extract the seperate state variables from an augmented state matrix. Calculate the mean and covariance, and resample
    this.

    Parameters
    ----------
    aug_upd_state : ndarray
        Augmented matrix of state variables
    state : dict
        Dict. af state variables
    list_state : list
        List of state variable
    new_en_size : int
        Size of the new ensemble

    Returns
    -------
    state : dict
        Dict. of resampled members
    """

    aug_row = 0
    curr_ne = state[list_state[0]].shape[1]
    new_state = {}
    for elem in list_state:
        # determine how many rows to extract
        no_rows = state[elem].shape[0]
        new_state[elem] = np.empty((no_rows, new_en_size))

        mean_state = np.mean(aug_state[aug_row:aug_row + no_rows, :], 1)
        pert_state = np.sqrt(1/(curr_ne - 1)) * (aug_state[aug_row:aug_row + no_rows, :] - np.dot(np.resize(mean_state,
                                                                                                            (len(mean_state), 1)), np.ones((1, curr_ne))))
        for i in range(new_en_size):
            new_state[elem][:, i] = mean_state + \
                np.dot(pert_state, np.random.normal(0, 1, pert_state.shape[1]))

        aug_row += no_rows

    return new_state


def block_diag_cov(cov, list_state):
    """
    Block diagonalize a covariance matrix dictionary.

    Parameters
    ----------
    cov : dict
        Dict. with cov. matrices
    list_state : list
        Fixed list of keys in state dict.

    Returns
    -------
    cov_out : ndarray
        Block diag. matrix with prior covariance matrices for each state.
    """
    # TODO: Change if there are cross-correlation between different states

    # Init. block in matrix
    cov_out = cov[list_state[0]]

    # Test if scalar has been given in init. block
    if not hasattr(cov_out, '__len__'):
        cov_out = np.array([[cov_out]])

    # Loop of rest of the state-names and add in block diag. matrix
    for i in range(1, len(list_state)):
        cov_out = linalg.block_diag(cov_out, cov[list_state[i]])

    # Return
    return cov_out


def calc_kalman_filter_eq(aug_state, kalman_gain, obs_data, pred_data):
    """
    Calculate the updated augment state using the Kalman filter equations

    Parameters
    ----------
    aug_state : ndarray
        Augmented state variable (all the parameters defined in `STATICVAR` augmented in one array)
    kalman_gain : ndarray
        Kalman gain
    obs_data : ndarray
        Augmented observed data vector (all `OBSNAME` augmented in one array)
    pred_data : ndarray
        Augmented predicted data vector (all `OBSNAME` augmented in one array)

    Returns
    -------
    aug_state_upd : ndarray
        Updated augmented state variable using the Kalman filter equations
    """
    # TODO: Implement svd updating algorithm

    # Matrix version
    # aug_state_upd = aug_state + np.dot(kalman_gain, (obs_data - pred_data))

    # For-loop version
    aug_state_upd = np.zeros(aug_state.shape)  # Init. updated state

    for i in range(aug_state.shape[1]):  # Loop over ensemble members
        aug_state_upd[:, i] = aug_state[:, i] + \
            np.dot(kalman_gain, (obs_data[:, i] - pred_data[:, i]))

    # Return the updated state
    return aug_state_upd


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
            state[var][state[var] < prior_info[var]['limits']
                       [0][0]] = prior_info[var]['limits'][0][0]
            state[var][state[var] > prior_info[var]['limits']
                       [0][1]] = prior_info[var]['limits'][0][1]
    return state


def subsample_state(index, aug_state, pert_state):
    """
    Draw a subsample from the original state, given by the index

    Parameters
    ----------
    index : ndarray
        Index of parameters to draw.
    aug_state : ndarray
        Original augmented state.
    pert_state : ndarray
        Perturbed augmented state, for error covariance.

    Returns
    -------
    new_state : dict
        Subsample of state.
    """

    new_state = np.empty((aug_state.shape[0], len(index)))
    for i in range(len(index)):
        new_state[:, i] = aug_state[:, index[i]] + \
            np.dot(pert_state, np.random.normal(0, 1, pert_state.shape[1]))
        # select some elements

    return new_state


def init_local_analysis(init, state):
    """Initialize local analysis.

    Initialize the local analysis by reading the input variables, defining the parameter classes and search ranges. Build
    the map of data/parameter positions.

    Args : 
        init : dictionary containing the parsed information form the input file.
        state : list of states that will be updated
    Returns : 
        local : dictionary of initialized values.
    """

    local = {}
    local['cell_parameter'] = []
    local['region_parameter'] = []
    local['vector_region_parameter'] = []
    local['unique'] = True

    for i, opt in enumerate(list(zip(*init))[0]):
        if opt.lower() == 'region_parameter':  # define scalar parameters valid in a region
            local['region_parameter'] = [
                elem for elem in init[i][1].split(' ') if elem in state]
        if opt.lower() == 'vector_region_parameter': # Sometimes it useful to define the same parameter for multiple
                                                    # regions as a vector.
            local['vector_region_parameter'] = [
                elem for elem in init[i][1].split(' ') if elem in state]
        if opt.lower() == 'cell_parameter':  # define cell specific vector parameters
            local['cell_parameter'] = [
                elem for elem in init[i][1].split(' ') if elem in state]
        if opt.lower() == 'search_range':
            local['search_range'] = int(init[i][1])
        if opt.lower() == 'column_update':
            local['column_update'] = [elem for elem in init[i][1].split(',')]
        if opt.lower() == 'parameter_position_file':  # assume pickled format
            with open(init[i][1], 'rb') as file:
                local['parameter_position'] = pickle.load(file)
        if opt.lower() == 'data_position_file':  # assume pickled format
            with open(init[i][1], 'rb') as file:
                local['data_position'] = pickle.load(file)
        if opt.lower() == 'update_mask_file':
            with open(init[i][1], 'rb') as file:
                local['update_mask'] = pickle.load(file)

    if 'update_mask' in local:
        return local
    else:
        assert 'parameter_position' in local, 'A pickle file containing the binary map of the parameters is MANDATORY'
        assert 'data_position' in local, 'A pickle file containing the position of the data is MANDATORY'

        data_name = [elem for elem in local['data_position'].keys()]
        if type(local['data_position'][data_name[0]][0]) == list:  # assim index has spesific position
            local['unique'] = False
            data_pos = [elem for data in data_name for assim_elem in local['data_position'][data]
                        for elem in assim_elem]
            data_ind = [f'{data}_{assim_indx}' for data in data_name for assim_indx, assim_elem in enumerate(local['data_position'][data])
                        for _ in assim_elem]
        else:
            data_pos = [elem for data in data_name for elem in local['data_position'][data]]
            # store the name for easy index
            data_ind = [data for data in data_name for _ in local['data_position'][data]]
        kde_search = cKDTree(data=data_pos)

        local['update_mask'] = {}
        for param in local['cell_parameter']:  # find data in a distance from the parameter
            field_size = local['parameter_position'][param].shape
            local['update_mask'][param] = [[[[] for _ in range(field_size[2])] for _ in range(field_size[1])] for _
                                           in range(field_size[0])]
            for k in range(field_size[0]):
                for j in range(field_size[1]):
                    new_iter = [elem for elem, val in enumerate(
                        local['parameter_position'][param][k, j, :]) if val]
                    if len(new_iter):
                        for i in new_iter:
                            local['update_mask'][param][k][j][i] = set(
                                [data_ind[elem] for elem in kde_search.query_ball_point(x=(k, j, i),
                                                                                        r=local['search_range'], workers=-1)])

        # see if data is inside the region. Note parameter_position is boolean map
        for param in local['region_parameter']:
            in_region = [local['parameter_position'][param][elem] for elem in data_pos]
            local['update_mask'][param] = set(
                [data_ind[count] for count, val in enumerate(in_region) if val])

        return local
