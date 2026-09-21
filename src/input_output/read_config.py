"""Parse config files."""
from input_output.config import is_dataassim, normalize as normalize_config
from pathlib import Path
import tomli
import tomli_w
import yaml
from yaml.loader import FullLoader
import numpy as np
import os


def read(filename: str):
    ''' Read configuration file. Supported formats are toml, .yaml, .pipt and .popt.'''
    if Path(filename).suffix.lower() == ".toml":
        return read_toml(filename)
    elif Path(filename).suffix.lower() in [".yaml", ".yml"]:
        return read_yaml(filename)
    elif Path(filename).suffix.lower() in [".pipt", ".popt"]:
        return read_txt(filename)
    else:
        raise ValueError('File format not supported. Supported formats are toml, .yaml, .pipt, .popt')


def read_yaml(filepath: str):
    """
    Read and parse a .yaml configuration file for PIPT/POPT.

    The YAML file should contain one or more of the following top-level keys:
    - 'dataassim' (dict): Data assimilation configuration
    - 'optim' (dict): Optimization configuration
    - 'fwdsim' (dict): Forward simulation configuration
    - 'ensemble' (dict, optional): Ensemble configuration

    Returns
    -------
    tuple
        (keys_pr, keys_fwd, keys_en)
        - keys_pr: dict, parsed 'dataassim' or 'optim' section (empty if not present)
        - keys_fwd: dict, parsed 'fwdsim' section (empty if not present)
        - keys_en: dict, parsed 'ensemble' section (empty if not present)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the YAML file is missing required sections.
    yaml.YAMLError
        If the YAML file is invalid.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"YAML file '{filepath}' does not exist.")

    # Register a custom constructor for !ndarray if needed
    def ndarray_constructor(loader, node):
        array = loader.construct_sequence(node)
        return np.array(array)
    yaml.add_constructor('!ndarray', ndarray_constructor)

    with open(filepath, "rb") as f:
        try:
            config = yaml.load(f, Loader=FullLoader)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file '{filepath}': {e}")

    if not isinstance(config, dict):
        raise ValueError(f"YAML file '{filepath}' does not contain a valid dictionary at the top level.")

    # Extract sections
    cfg_ens = config.get("ensemble", {})
    cfg_sim = config.get("fwdsim") or config.get("simulator") or {}
    cfg_prb = config.get("dataassim") or config.get("optim") or {}

    return normalize_config(cfg_prb, cfg_sim, cfg_ens)


def read_toml(filepath: str):
    """
    Read and parse a .toml configuration file for PIPT/POPT.

    The TOML file should contain one or more of the following top-level keys:
    - 'dataassim' (dict): Data assimilation configuration
    - 'optim' (dict): Optimization configuration
    - 'fwdsim' (dict): Forward simulation configuration
    - 'ensemble' (dict, optional): Ensemble configuration

    Returns
    -------
    tuple
        (keys_pr, keys_fwd, keys_en)
        - keys_pr: dict, parsed 'dataassim' or 'optim' section (empty if not present)
        - keys_fwd: dict, parsed 'fwdsim' section (empty if not present)
        - keys_en: dict, parsed 'ensemble' section (empty if not present)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the TOML file is missing required sections.
    tomli.TOMLDecodeError
        If the TOML file is invalid.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"TOML file '{filepath}' does not exist.")

    with open(filepath, 'rb') as f:
        try:
            config = tomli.load(f)
        except tomli.TOMLDecodeError as e:
            raise tomli.TOMLDecodeError(f"Error parsing TOML file '{filepath}': {e}")

    if not isinstance(config, dict):
        raise ValueError(f"TOML file '{filepath}' does not contain a valid dictionary at the top level.")

    # Extract sections
    cfg_ens = config.get("ensemble", {})
    cfg_sim = config.get("fwdsim") or config.get("simulator") or {}
    cfg_prb = config.get("dataassim") or config.get("optim") or {}

    return normalize_config(cfg_prb, cfg_sim, cfg_ens)


def convert_txt_to_toml(init_file):
    """Write a legacy ``.pipt``/``.popt`` file as ``<name>.toml`` next to it."""
    # Read .pipt or .popt file
    pr, fwd, _ = read_txt(init_file)

    # Write dictionaries to toml file with same base file name
    new_file = change_file_extension(init_file, 'toml')
    with open(new_file, 'wb') as f:
        if is_dataassim(pr):
            tomli_w.dump({'dataassim': pr, 'fwdsim': fwd}, f)
        else:
            tomli_w.dump({'optim': pr, 'fwdsim': fwd}, f)

def convert_txt_to_yaml(init_file):
    """Write a legacy ``.pipt``/``.popt`` file as ``<name>.yaml`` next to it."""
    # Read .pipt or .popt file
    pr, fwd, _ = read_txt(init_file)

    # Write dictionaries to yaml file with same base file name
    new_file = change_file_extension(init_file, 'yaml')
    with open(new_file, 'w') as f:
        if is_dataassim(pr):
            yaml.dump({'dataassim': pr, 'fwdsim': fwd}, f)
        else:
            yaml.dump({'optim': pr, 'fwdsim': fwd}, f)

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
    keys_pr = parse_keywords(clean_lines_pr) if pr_part in ('dataassim', 'optim') else None
    keys_fwd = parse_keywords(clean_lines_fwd)
    # Three sections, like the other readers; the text format keeps the
    # ensemble's keys in DATAASSIM, so the third is empty. What is missing is
    # reported by `pet validate` and when the run is built, not asserted here.
    return normalize_config(keys_pr, keys_fwd, None)


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


def _coerce_keyword_rows(rows):
    """
    Convert the raw text rows following a keyword into a typed value.

    ``rows`` is a list of the raw (whitespace/tab-separated) strings that
    followed a keyword in the init. file. Depending on how many rows there
    are, and whether their tokens parse as numbers, the result is a float or
    string scalar, a 1D list, or a 2D list. Numeric parsing is attempted
    first (scalar, then 1D, then 2D); if that fails at every level the value
    is treated as string data instead.
    """
    if len(rows) == 1:
        row = rows[0]
        if len(row.split()) == 1:
            try:
                return float(row)
            except Exception:
                pass
        try:
            return [float(x) for x in row.split()]
        except Exception:
            pass
        tokens = row.split('\t')
        if len(tokens) == 1:
            return row.strip().lower()
        return [x.rstrip('\n').lower() for x in tokens if x != '']

    # Multiple rows: try a flat 1D float list (one float per row) first...
    try:
        return [float(x) for x in rows]
    except Exception:
        pass

    # ...then a 2D float list (each row is one or more whitespace-separated floats)...
    try:
        return [[float(x) for x in col.split()] for col in rows]
    except Exception:
        pass

    # ...and finally fall back to string data: one column per row becomes a 1D
    # list of strings, multiple (tab-separated) columns become a 2D list.
    one_col = all(len(row.split('\t')) == 1 for row in rows)
    if one_col:
        return [x.rstrip('\n').lower() for x in rows]
    return [[x.rstrip('\n').lower() for x in col.split('\t') if x != ''] for col in rows]


def _promote_token(token):
    """Convert a string token to a float or list of floats where possible, else leave it unchanged."""
    try:
        return float(token)
    except Exception:
        pass
    try:
        return [float(x) for x in token.split()]
    except Exception:
        return token


def _promote_numeric_strings(keys):
    """
    Retroactively convert list values that were parsed as pure strings back to
    numbers, where every entry (or sub-entry) actually parses as a float.

    ``_coerce_keyword_rows`` only recognizes a row block as numeric if *all*
    of its rows parse as floats, so a keyword with a mix of numeric and
    string rows ends up stored as strings. This fixes up such keywords
    entry-by-entry after the fact.
    """
    for value in keys.values():
        if not isinstance(value, list):
            continue
        if isinstance(value[0], list):
            for row in value:
                if all(isinstance(x, str) for x in row):
                    row[:] = [_promote_token(x) for x in row]
        elif all(isinstance(x, str) for x in value):
            value[:] = [_promote_token(x) for x in value]


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
    keys = {}
    for line in lines:
        if not line:  # Empty list corresponds to an empty line in the file
            continue
        keyword = line[0].strip().lower()
        keys[keyword] = _coerce_keyword_rows(line[1:])

    _promote_numeric_strings(keys)
    return keys


def change_file_extension(filename, new_extension):
    """``filename`` with its extension replaced by ``new_extension``."""
    if '.' in filename:
        name, old_extension = filename.rsplit('.', 1)
        new_filename = name + '.' + new_extension
    else:
        new_filename = filename + '.' + new_extension
    return new_filename
