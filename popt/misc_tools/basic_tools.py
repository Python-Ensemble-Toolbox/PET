"""
Collection of simple, yet useful Python tools
"""


import numpy as np
import sys

def index2d(list2d, value):
    """
    Search in a 2D list for pattern or value and return is (i, j) index. If the
    pattern/value is not found, (None, None) is returned

    Examples
    --------

    >>> l = [['string1', 1], ['string2', 2]]
    >>> print index2d(l, 'string1')
    (0, 0)

    Parameters
    ----------
    list2d : list of lists
        2D list.

    value : object
        Pattern or value to search for.

    Returns
    -------
    ind : tuple
        Indices (i, j) of the value.
    """
    return next(((i, j) for i, lst in enumerate(list2d) for j, x in enumerate(lst) if x == value), None)


def read_file(val_type, filename):
    """
    Read an eclipse file with specified keyword.

    Examples
    --------
    >>> read_file('PERMX','filename.permx')

    Parameters
    ----------
    val_type :
        keyword or property
    filename :
        the file that is read

    Returns
    -------
    values :
        a vector with values for each cell
    """

    file = open(filename, 'r')
    lines = file.readlines()
    key = ''
    line_idx = 0
    while key != val_type:
        line = lines[line_idx]
        if not line:
            print('Error: Keyword not found')
            sys.exit(1)

        line_idx += 1
        if len(line):
            key = line.split()
            if key:
                key = key[0]
    data = []
    finished = False
    while line_idx < len(lines) and not finished:
        line = lines[line_idx]
        line_idx += 1
        if line == '\n' or line[:2] == '--':
            continue
        if line == '':
            break
        if line.strip() == '/':
            finished = True
        sub_str = line.split()
        for s in sub_str:
            if '*' in s:
                num_val = s.split('*')
                v = float(num_val[1]) * np.ones(int(num_val[0]))
                data.append(v)
            elif '/' in s:
                finished = True
                break
            else:
                data.append(float(s))

    values = np.hstack(data)
    return values


def write_file(filename, val_type, data):
    """Write an eclipse file with specified keyword.

    Examples
    --------
    >>> write_file('filename.permx','PERMX',data_vec)

    Parameters
    ----------
    filename :
        the file that is read
    val_type:
        keyword or property
    data :
        data written to file
    """

    file = open(filename, 'w')
    file.writelines(val_type + '\n')
    if data.dtype == 'int64':
        np.savetxt(file, data, fmt='%i')
    else:
        np.savetxt(file, data)
    file.writelines('/' + '\n')
    file.close()
