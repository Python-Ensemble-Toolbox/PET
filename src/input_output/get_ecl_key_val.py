"""Descriptive description."""

import numpy as np
import sys


def read_file(val_type, filename):

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

    file = open(filename, 'w')
    file.writelines(val_type + '\n')
    if data.dtype == 'int64':
        np.savetxt(file, data, fmt='%i')
    else:
        np.savetxt(file, data)
    file.writelines('/' + '\n')
    file.close()
