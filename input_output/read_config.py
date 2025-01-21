"""Parse config files."""
from misc import read_input_csv as ricsv
from copy import deepcopy
from input_output.organize import Organize_input
import tomli
import tomli_w
import yaml
from yaml.loader import FullLoader
import numpy as np


def convert_txt_to_yaml(init_file):
    # Read .pipt or .popt file
    pr, fwd = read_txt(init_file)

    # Write dictionaries to yaml file with same base file name
    new_file = change_file_extension(init_file, 'yaml')
    with open(new_file, 'wb') as f:
        if 'daalg' in pr:
            yaml.dump({'dataassim': pr, 'fwdsim': fwd}, f)
        else:
            yaml.dump({'optim': pr, 'fwdsim': fwd}, f)


def read_yaml(init_file):
    """
    Read .yaml input file, parse and return dictionaries for PIPT/POPT.

    Parameters
    ----------
    init_file : str
        .yaml file

    Returns
    -------
    keys_da : dict
        Parsed keywords from dataassim
    keys_fwd : dict
        Parsed keywords from fwdsim
    """
    # Make a !ndarray tag to convert a sequence to np.array
    def ndarray_constructor(loader, node):
        array = loader.construct_sequence(node)
        return np.array(array)

    # Add constructor to yaml with tag !ndarray
    yaml.add_constructor('!ndarray', ndarray_constructor)

    # Read
    with open(init_file, 'rb') as fid:
        y = yaml.load(fid, Loader=FullLoader)

    # Check for dataassim and fwdsim
    if 'optim' in y.keys():
        keys_pr = y['optim']
        check_mand_keywords_opt(keys_pr)
    elif 'dataassim' in y.keys():
        keys_pr = y['datasssim']
        check_mand_keywords_da(keys_pr)
    else:
        raise KeyError
    if 'fwdsim' in y.keys():
        keys_fwd = y['fwdsim']
    else:
        raise KeyError

    # Organize keywords
    org = Organize_input(keys_pr, keys_fwd)
    org.organize()

    return org.get_keys_pr(), org.get_keys_fwd()


def convert_txt_to_toml(init_file):
    # Read .pipt or .popt file
    pr, fwd = read_txt(init_file)

    # Write dictionaries to toml file with same base file name
    new_file = change_file_extension(init_file, 'toml')
    with open(new_file, 'wb') as f:
        if 'daalg' in pr:
            tomli_w.dump({'dataassim': pr, 'fwdsim': fwd}, f)
        else:
            tomli_w.dump({'optim': pr, 'fwdsim': fwd}, f)


def read_toml(init_file):
    """
    Read .toml configuration file, parse and output dictionaries for PIPT/POPT

    Parameters
    ----------
    init_file : str
        toml configuration file
    """
    # Read
    with open(init_file, 'rb') as fid:
        t = tomli.load(fid)

    # Check for dataassim and fwdsim
    if 'ensemble' in t.keys():
        keys_en = t['ensemble']
        check_mand_keywords_en(keys_en)
    else:
        keys_en = None
    if 'optim' in t.keys():
        keys_pr = t['optim']
        check_mand_keywords_opt(keys_pr)
    elif 'dataassim' in t.keys():
        keys_pr = t['dataassim']
        check_mand_keywords_da(keys_pr)
    else:
        raise KeyError
    if 'fwdsim' in t.keys():
        keys_fwd = t['fwdsim']
    else:
        raise KeyError

    # Organize keywords
    org = Organize_input(keys_pr, keys_fwd, keys_en)
    org.organize()

    return org.get_keys_pr(), org.get_keys_fwd(), org.get_keys_en()


def read_txt(init_file):
    """
    Read a PIPT or POPT input file (.pipt or .popt), parse and output dictionaries for data assimilation or
    optimization,  and simulator classes.

    Parameters
    ----------
    init_file : str
        PIPT init. file containing info. to run the inversion algorithm

    Returns
    -------
    keys_pr : dict
        Parsed keywords from DATAASSIM or OPTIM
    keys_fwd : dict
        Parsed keywords from FWDSSIM
    """

    # Check for .pipt suffix
    if not init_file.endswith('.pipt') and not init_file.endswith('.popt'):
        raise FileNotFoundError(f'No PIPT or POPT input file (.pipt or .popt) found! If {init_file} is  '
                                f'a PIPT or POPT input file, change suffix to .pipt or .popt')

    # Read the init file and output lines without comments (lines starting with '#')
    lines = read_clean_file(init_file)

    # Find where the separate parts are located in the file. FWDSIM will always be a part, but the
    # inversion/optimiztation part may be DATAASSIM or OPTIM
    prind = None
    pr_part = None
    fwdsimind = None
    for i in range(len(lines)):
        if lines[i].strip().lower() == 'dataassim' or lines[i].strip().lower() == 'optim':
            prind = i
            pr_part = lines[i].strip().lower()
        elif lines[i].strip().lower() == 'fwdsim':
            fwdsimind = i

    # Split the file into the two separate parts. Each part will (only) contain the keywords of each part:
    if prind < fwdsimind:  # Data assim. part is the first part of file
        lines_pr = lines[2:fwdsimind]
        lines_fwd = lines[fwdsimind + 2:]
    else:  # Fwd sim. part is the first part of file
        lines_fwd = lines[2:prind]
        lines_pr = lines[prind + 2:]

    # Get rid of empty lines in lines_pr and lines_fwd
    clean_lines_pr = remove_empty_lines(lines_pr)
    clean_lines_fwd = remove_empty_lines(lines_fwd)

    # Assign the keys and values to different dictionaries depending on whether we have data assimilation (DATAASSIM)
    # or optimization (OPTIM). FWDSIM info is always assigned to keys_fwd
    keys_pr = None
    if pr_part == 'dataassim':
        keys_pr = parse_keywords(clean_lines_pr)
        check_mand_keywords_da(keys_pr)
    elif pr_part == 'optim':
        keys_pr = parse_keywords(clean_lines_pr)
        check_mand_keywords_opt(keys_pr)
    keys_fwd = parse_keywords(clean_lines_fwd)
    check_mand_keywords_fwd(keys_fwd)

    org = Organize_input(keys_pr, keys_fwd)
    org.organize()

    return org.get_keys_pr(), org.get_keys_fwd()


def read_clean_file(init_file):
    """
    Read PIPT init. file and lines that are not comments (marked with octothorpe)

    Parameters
    ----------
    init_file : str
        Name of file to remove all comments. WHOLE filename needed (with suffix!)

    Returns
    -------
    lines : list
        Lines from init. file converted to list entries
    """
    # Read file except lines starting with an octothorpe (#) and return the python variable
    with open(init_file, 'r') as f:
        lines = [line for line in f.readlines() if not line.startswith('#')]

    # Return clean lines
    return lines


def remove_empty_lines(lines):
    """
    Small method for finding empty lines in a read file.

    Parameters
    ----------
    lines : list
        List of lines from a file

    Returns
    -------
    lines_clean : list
        List of clean lines (without empty entries)
    """
    # Loop over lines to find '\n'
    sep = []
    for i in range(len(lines)):
        if lines[i] == '\n':
            sep.append(i)

    # Make clean output
    lines_clean = []
    for i in range(len(sep)):
        if i == 0:
            lines_clean.append(lines[0:sep[i]])
        else:
            lines_clean.append(lines[sep[i-1] + 1:sep[i]])

    # Return
    return lines_clean


def parse_keywords(lines):
    """
    Here we parse the lines in the init. file to a Python dictionary. The keys of the dictionary is the keywords
    in the PIPT init. file, and the information in each keyword is stored in each key of the
    dictionary. To know how the keyword-information is organized in the keys of the dictionary, confront the
    manual located in the doc folder.

    Parameters
    ----------
    lines : list
        List of (clean) lines from the PIPT init. file.

    Returns
    -------
    keys : dict
        Dictionary with all info. from the init. file.
    """
    # Init. the dictionary
    keys = {}

    # Loop over all input keywords and store in the dictionary.
    for i in range(len(lines)):
        if lines[i] != []:  # Check for empty list (corresponds to empty line in file)
            try:  # Try first to store the info. in keyword as float in a 1D list
                # A scalar, which we store as scalar...
                if len(lines[i][1:]) == 1 and len(lines[i][1:][0].split()) == 1:
                    keys[lines[i][0].strip().lower()] = float(lines[i][1:][0])
                else:
                    keys[lines[i][0].strip().lower()] = [float(x) for x in lines[i][1:]]
            except:
                try:  # Store as float in 2D list
                    if len(lines[i][1:]) == 1:  # Check if it is actually a 1D array disguised as 2D
                        keys[lines[i][0].strip().lower()] = \
                            [float(x) for x in lines[i][1:][0].split()]
                    else:  # if not store as 2D list
                        keys[lines[i][0].strip().lower()] = \
                            [[float(x) for x in col.split()] for col in lines[i][1:]]
                except:  # Keyword contains string(s), not floats
                    if len(lines[i][1:]) == 1:  # If 1D list
                        # If it is a scalar store as single input
                        if len(lines[i][1:][0].split('\t')) == 1:
                            keys[lines[i][0].strip().lower()] = lines[i][1:][0].strip().lower()
                        else:  # Store as 1D list
                            keys[lines[i][0].strip().lower()] = \
                                [x.rstrip('\n').lower()
                                 for x in lines[i][1:][0].split('\t') if x != '']
                    else:  # It is a 2D list
                        # Check each row in 2D list. If it is single column (i.e., one string per row),
                        # we make it a 1D list of strings; if not, we make it a 2D list of strings.
                        one_col = True
                        for j in range(len(lines[i][1:])):
                            if len(lines[i][1:][j].split('\t')) > 1:
                                one_col = False
                                break
                        if one_col is True:  # Only one column
                            keys[lines[i][0].strip().lower()] = \
                                [x.rstrip('\n').lower() for x in lines[i][1:]]
                        else:  # Store as 2D list
                            keys[lines[i][0].strip().lower()] = \
                                [[x.rstrip('\n').lower() for x in col.split('\t') if x != '']
                                    for col in lines[i][1:]]

    # Need to check if there are any only-string-keywords that actually contains floats, and convert those to
    # floats (the above loop only handles pure float or pure string input, hence we do a quick fix for mixed
    # lists here)
    # Loop over all keys in dict. and check every "pure" string keys for floats
    for i in keys:
        if isinstance(keys[i], list):  # Check if key is a list
            if isinstance(keys[i][0], list):  # Check if it is a 2D list
                for j in range(len(keys[i])):  # Loop over all sublists
                    # Check sublist for strings
                    if all(isinstance(x, str) for x in keys[i][j]):
                        for k in range(len(keys[i][j])):  # Loop over enteries in sublist
                            try:  # Try to make float
                                keys[i][j][k] = float(keys[i][j][k])  # Scalar
                            except:
                                try:  # 1D array
                                    keys[i][j][k] = [float(x)
                                                     for x in keys[i][j][k].split()]
                                except:  # If it is actually a string, pass over
                                    pass
            else:  # It is a 1D list
                # Check if list only contains strings
                if all(isinstance(x, str) for x in keys[i]):
                    for j in range(len(keys[i])):  # Loop over all entries in list
                        try:  # Try to make float
                            keys[i][j] = float(keys[i][j])
                        except:
                            try:
                                keys[i][j] = [float(x) for x in keys[i][j].split()]
                            except:  # If it is actually a string, pass over
                                pass

    # Return dict.
    return keys


def check_mand_keywords_fwd(keys_fwd):
    """Check for mandatory keywords in `FWDSIM` part, and output error if they are not present"""

    # Mandatory keywords in FWDSIM
    assert 'parallel' in keys_fwd, 'PARALLEL not in FWDSIM!'
    assert 'datatype' in keys_fwd, 'DATATYPE not in FWDSIM!'


def check_mand_keywords_da(keys_da):
    """Check for mandatory keywords in `DATAASSIM` part, and output error if they are not present"""

    # Mandatory keywords in DATAASSIM
    assert 'truedataindex' in keys_da, 'TRUEDATAINDEX not in DATAASSIM!'
    assert 'assimindex' in keys_da, 'ASSIMINDEX not in DATAASSIM!'
    assert 'truedata' in keys_da, 'TRUEDATA not in DATAASSIM!'
    assert 'datavar' in keys_da, 'DATAVAR not in DATAASSIM!'
    assert 'obsname' in keys_da, 'OBSNAME not in DATAASSIM!'
    assert 'energy' in keys_da, 'ENERGY not in DATAASSIM!'


def check_mand_keywords_opt(keys_opt):
    """Check for mandatory keywords in `OPTIM` part, and output error if they are not present"""
pass


def check_mand_keywords_en(keys_en):
    """Check for mandatory keywords in `ENSEMBLE` part, and output error if they are not present"""

    # Mandatory keywords in ENSEMBLE
    assert 'ne' in keys_en, 'NE not in ENSEMBLE!'
    assert 'state' in keys_en, 'STATE not in ENSEMBLE!'
    if 'importstaticvar' not in keys_en:
        assert filter(list(keys_en.keys()),
                      'prior_*') != [], 'No PRIOR_<STATICVAR> in DATAASSIM'

def change_file_extension(filename, new_extension):
    if '.' in filename:
        name, old_extension = filename.rsplit('.', 1)
        new_filename = name + '.' + new_extension
    else:
        new_filename = filename + '.' + new_extension
    return new_filename
