# This module inlcudes fucntions for extracting information from input dicts
__all__ = [
    'extract_prior_info',
    'extract_multilevel_info',
    'extract_local_analysis_info',
    'extract_maxiter',
    'organize_sparse_representation',
    'list_to_dict'
]

# Imports 
import numpy as np
import pickle
import os

from scipy.spatial import cKDTree
from typing import Union


def extract_prior_info(keys: dict) -> dict:
    '''
    Extract prior information on STATE from keyword(s).
    '''
    # Get state names as list
    state_names = keys['state']
    if not isinstance(state_names, list): state_names = [state_names]

    # Check if PRIOR_<state names> exists for each entry in state
    for name in state_names:
        assert_msg = f'PRIOR_{name.upper()} is missing! This keyword is needed to make initial ensemble for {name.upper()} entered in STATE' 
        assert f'prior_{name}' in keys, assert_msg
    
    # Sefine dict to store prior information in 
    prior_info = {name: None for name in state_names}

    # loop over state priors
    for name in state_names:
        prior = keys[f'prior_{name}']
        
        # Check if is a list (old way)
        if isinstance(prior, list):
            prior = list_to_dict(prior)
        else:
            assert isinstance(prior, dict), f'PRIOR_{name.upper()} must be a dictionary or list of lists!'

        # Load mean if in file
        if 'mean' in prior:
            if isinstance(prior['mean'], str):
                assert prior['mean'].endswith('.npz'), 'File name does not end with \'.npz\'!'
                mean_file = np.load(prior['mean'])
                assert len(mean_file.files) == 1, \
                    f"More than one variable located in {prior['mean']}. Only the mean vector can be stored in the .npz file!" 
                prior['mean'] = mean_file[mean_file.files[0]]
            else:  # Single number inputted, make it a list if not already
                if not isinstance(prior['mean'], list):
                    prior['mean'] = [prior['mean']]
        else:
            prior['mean'] = [None]
       
        # loop over keys in prior
        for key in prior.keys():
            # ensure that entry is a list
            if (not isinstance(prior[key], list)) and (key != 'mean') and (key != 'active'):
                prior[key] = [prior[key]]

        # change the name of some keys
        if 'var' in prior:
            prior['variance'] = prior.pop('var', None)
        if 'range' in prior:
            prior['corr_length'] = prior.pop('range', None)

        # process grid
        if 'grid' in prior:
            grid_dim = prior['grid']

            # check if 3D-grid
            if (len(grid_dim) == 3) and (grid_dim[2] > 1):
                nz = int(grid_dim[2])
                prior['nz'] = nz
                prior['nx'] = int(grid_dim[0])
                prior['ny'] = int(grid_dim[1])
                

                # Check mean when values have been inputted directly (not when mean has been loaded)
                mean = prior['mean']
                if isinstance(mean, list) and len(mean) < nz:
                        # Check if it is more than one entry and give error
                    assert len(mean) == 1, \
                        'Information from MEAN has been given for {0} layers, whereas {1} is needed!' \
                        .format(len(mean), nz)

                    # Only 1 entry; copy this to all layers
                    print(
                        '\033[1;33mSingle entry for MEAN will be copied to all {0} layers\033[1;m'.format(nz))
                    prior['mean'] = mean * nz

                #check if info. has been given on all layers. In the case it has not been given, we just copy the info. given.
                for key in ['vario', 'variance', 'aniso', 'angle', 'corr_length']:
                    if key in prior.keys():
                        val = prior[key]
                        if len(val) < nz:
                            # Check if it is more than one entry and give error
                            assert len(val) == 1, \
                                'Information from {0} has been given for {1} layers, whereas {2} is needed!' \
                                .format(key.upper(), len(val), nz)

                            # Only 1 entry; copy this to all layers
                            print(
                                '\033[1;33mSingle entry for {0} will be copied to all {1} layers\033[1;m'.format(key.upper(), nz))
                            prior[key] = val * nz

            else:
                prior['nx'] = int(grid_dim[0])
                prior['ny'] = int(grid_dim[1])
                prior['nz'] = 1

        prior.pop('grid', None)

        # add prior to prior_info
        prior_info[name] = prior
        
    return prior_info


def extract_multilevel_info(keys: Union[dict, list]) -> dict:
    '''
    Extract the info needed for ML simulations. Note if the ML keyword is not in keys_en we initialize
    such that we only have one level -- the high fidelity one
    '''
    ml_info = keys['multilevel']
    if isinstance(ml_info, list):
        ml_info = list_to_dict(ml_info)
    assert isinstance(ml_info, dict)
    
    # Set levels
    levels = int(ml_info['levels'])
    ml_info['levels'] = [elem for elem in range(levels)]

    # Set multi-level ensemble size
    en_size = ml_info.pop('en_size')
    ml_info['ne'] = [range(int(elem)) for elem in en_size]
    ml_ne = [int(elem) for elem in en_size]

    # Set multi-level error
    if not 'ml_error_corr' in ml_info:
        ml_error_corr = 'none'
        error_comp_scheme = 'none'
        ml_corr_done = True
    else:
        ml_error_corr = ml_info['ml_error_corr'][0]
        ml_corr_done = False

    if not ml_error_corr == 'none':
        error_comp_scheme = ml_info['ml_error_corr'][1]

    # set attribute
    return ml_info, levels, ml_ne, ml_error_corr, error_comp_scheme, ml_corr_done 


def extract_local_analysis_info(keys: Union[dict, list], state: list) -> dict:
    # Check if keys are list, and make it a dict if not
    if isinstance(keys, list):
        keys = list_to_dict(keys)
    assert isinstance(keys, dict)

    # Initialize local dict
    local = {
        'cell_parameter': None,
        'region_parameter': None,
        'vector_region_parameter': None,
        'unique': True
    }

    # Loop over keys and fill in local
    for key, key_item in keys.items():
        if key.lower() in ['region_parameter', 'vector_region_parameter', 'cell_parameter']:
            local[key] = [elem for elem in key_item.split(' ') if elem in state]
        elif key.lower() == 'search_range':
            local[key] = int(key_item) 
        elif key.lower() == 'column_update':
            local[key] = [elem for elem in key_item.split(',')]
        elif key.lower().endswith('_file'): # 'parameter_position_file', 'data_position_file' or 'update_mask_file'
            with open(key_item, 'rb') as file:
                local[key.lower().strip('_file')] = pickle.load(file) # assume pickle format

    # Ensure that update_mask is there
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
    

def organize_sparse_representation(info: Union[dict,list]) -> dict:
    """
    Function for reading input to wavelet sparse representation of data.

    This function takes a dictionary (or a list convertible to a dictionary) describing
    the configuration for wavelet sparse representation, standardizes boolean options
    (interpreting 'yes'/'no' as True/False), loads or creates mask files, and collects
    all relevant parameters into a new dictionary suitable for downstream processing.

    Parameters
    ----------
    info : dict or list
        Input configuration for sparse representation. If a list, it will be converted
        to a dictionary. Expected keys include:
            - 'dim': list of 3 ints, the dimensions of the data grid.
            - 'mask': list of filenames for mask arrays.
            - 'level', 'wname', 'threshold_rule', 'th_mult', 'order', 'min_noise',
              'colored_noise', 'use_hard_th', 'keep_ca', 'inactive_value', 'use_ensemble'.

    Returns
    -------
    sparse : dict
        Dictionary containing the processed sparse representation configuration,
        with masks loaded or created, dimensions flipped for compatibility, and
        all options standardized.
    """
    # Ensure a dict
    if isinstance(info, list):
        info = list_to_dict(info)
    assert isinstance(info, dict)

    # Redefine all 'yes' and 'no' values to bool
    for key, val in info.items():
        if val == 'yes': info[key] == True
        if val == 'no':  info[key] == False

    # Intial dict
    sparse = {}

    # Flip dim to align with flow/eclipse
    dim = [int(x) for x in info['dim']]
    sparse['dim'] = [dim[2], dim[1], dim[0]]

    # Read mask_files
    sparse['mask'] = [] 
    for idx, filename in enumerate(info['mask'], start=1):
        if not os.path.exists(filename):
            mask = np.ones(sparse['dim'], dtype=bool)
            np.savez(f'mask_{idx}.npz', mask=mask)
        else:
            mask = np.load(filename)['mask']
        sparse['mask'].append(mask.flatten())

    # Read rest of keywords
    sparse['level'] = info['level']
    sparse['wname'] = info['wname']
    sparse['threshold_rule'] = info['threshold_rule']
    sparse['th_mult'] = info['th_mult']
    sparse['order'] = info['order']
    sparse['min_noise'] = info['min_noise']
    sparse['colored_noise'] = info.get('colored_noise', False)
    sparse['use_hard_th'] = info.get('use_hard_th', False)
    sparse['keep_ca'] = info.get('keep_ca', False)
    sparse['inactive_value'] = info['inactive_value']
    sparse['use_ensemble'] = info.get('use_ensemble', None)

    return sparse


def extract_maxiter(keys: dict) -> dict:

    if 'iteration' in keys:
        if isinstance(keys['iteration'], list): 
            keys['iteration'] = list_to_dict(keys['iteration'])
        try:
            max_iter = keys['iteration']['max_iter']
        except KeyError:
                raise AssertionError('MAX_ITER has not been given in ITERATION')
        
    elif 'mda' in keys:
        if isinstance(keys['mda'], list): 
            keys['mda'] = list_to_dict(keys['mda'])
        try:
            max_iter = keys['mda']['max_iter']
        except KeyError:
                raise AssertionError('MAX_ITER has not been given in MDA')

    else:
        max_iter = 1

    return max_iter
    
   
def list_to_dict(info_list: list) -> dict:
    assert isinstance(info_list, list)
    # Initialize and loop over entries
    info_dict = {}
    for entry in info_list:
        if not isinstance(entry, list):
            entry = [entry]
        # Fill in values
        if len(entry) == 1:
            info_dict[str(entry[0])] = None
        elif len(entry) == 2:
            info_dict[str(entry[0])] = entry[1]
        else:
            info_dict[str(entry[0])] = entry[1:]

    return info_dict
