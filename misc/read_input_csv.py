"""
File for reading CSV files and returning a 2D list
"""
import pandas as pd
import numpy as np


def read_data_csv(filename, datatype, truedataindex):
    """
    Parameters
    ----------
    filename:
        Name of csv-file
    datatype:
        List of data types as strings
    truedataindex:
        List of where the "TRUEDATA" has been extracted (e.g., at which time, etc)

    Returns
    -------
    some-type:
        List of observed data
    """

    df = pd.read_csv(filename)  # Read the file

    imported_data = []  # Initialize the 2D list of csv data
    tlength = len(truedataindex)
    dnumber = len(datatype)

    if df.columns[0] == 'header_both':  # csv file has column and row headers
        pos = [None] * dnumber
        for col in range(dnumber):
            # find index of data type in csv file header
            pos[col] = df.columns.get_loc(datatype[col])
        for t in truedataindex:
            row = df[df['header_both'] == t]  # pick row corresponding to truedataindex
            row = row.values[0]  # select the values of the dataframe row
            csv_data = [None] * dnumber
            for col in range(dnumber):
                if (not type(row[pos[col]]) == str) and (np.isnan(row[pos[col]])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(row[pos[col]])
                    except:  # It is a string
                        csv_data[col] = row[pos[col]]
            imported_data.append(csv_data)
    else:  # No row headers (the rows in the csv file must correspond to the order in truedataindex)
        if tlength == df.shape[0]:  # File has column headers
            pos = [None] * dnumber
            for col in range(dnumber):
                # Find index of the header in datatype
                pos[col] = df.columns.get_loc(datatype[col])
        # File has no column headers (columns must correspond to the order in datatype)
        elif tlength == df.shape[0]+1:
            # First row has been misinterpreted as header, so we read first row again:
            temp = pd.read_csv(filename, header=None, nrows=1).values[0]
            pos = list(range(df.shape[1]))  # Assume the data is in the correct order
            csv_data = [None] * len(temp)
            for col in range(len(temp)):
                if (not type(temp[col]) == str) and (np.isnan(temp[col])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(temp[col])
                    except:  # It is a string
                        csv_data[col] = temp[col]
            imported_data.append(csv_data)

        for rows in df.values:
            csv_data = [None] * dnumber
            for col in range(dnumber):
                if (not type(rows[pos[col]]) == str) and (np.isnan(rows[pos[col]])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(rows[pos[col]])
                    except:  # It is a string
                        csv_data[col] = rows[pos[col]]
            imported_data.append(csv_data)

    return imported_data


def read_var_csv(filename, datatype, truedataindex):
    """
    Parameters
    ----------
    filename : str
        Name of the CSV file.

    datatype : list
        List of data types as strings.

    truedataindex : list
        List of indices where the "TRUEDATA" has been extracted.

    Returns
    -------
    imported_var : list
        List of variances.
    """

    df = pd.read_csv(filename)  # Read the file

    imported_var = []  # Initialize the 2D list of csv data
    tlength = len(truedataindex)
    dnumber = len(datatype)

    if df.columns[0] == 'header_both':  # csv file has column and row headers
        pos = [None] * dnumber
        for col in range(dnumber):
            # find index of data type in csv file header
            pos[col] = df.columns.get_loc(datatype[col])
        for t in truedataindex:
            row = df[df['header_both'] == t]  # pick row
            row = row.values[0]  # select the values of the dataframe
            csv_data = [None] * 2 * dnumber
            for col in range(dnumber):
                csv_data[2*col] = row[pos[col]]
                try:  # Making a float
                    csv_data[2*col+1] = float(row[pos[col]]+1)
                except:  # It is a string
                    csv_data[2*col+1] = row[pos[col]+1]
            # Make sure the string input is lowercase
            csv_data[0::2] = [x.lower() for x in csv_data[0::2]]
            imported_var.append(csv_data)
    else:  # No row headers (the rows in the csv file must correspond to the order in truedataindex)
        if tlength == df.shape[0]:  # File has column headers
            pos = [None] * dnumber
            for col in range(dnumber):
                # Find index of datatype in csv file header
                pos[col] = df.columns.get_loc(datatype[col])
        # File has no column headers (columns must correspond to the order in datatype)
        elif tlength == df.shape[0]+1:
            # First row has been misinterpreted as header, so we read first row again:
            temp = pd.read_csv(filename, header=None, nrows=1).values[0]
            # Make sure the string input is lowercase
            temp[0::2] = [x.lower() for x in temp[0::2]]
            # Assume the data is in the correct order
            pos = list(range(0, df.shape[1], 2))
            csv_data = [None] * len(temp)
            for col in range(dnumber):
                csv_data[2 * col] = temp[2 * col]
                try:  # Making a float
                    csv_data[2*col+1] = float(temp[2*col+1])
                except:  # It is a string
                    csv_data[2*col+1] = temp[2*col+1]
            imported_var.append(csv_data)

        for rows in df.values:
            csv_data = [None] * 2 * dnumber
            for col in range(dnumber):
                csv_data[2*col] = rows[2*col]
                try:  # Making a float
                    csv_data[2*col+1] = float(rows[pos[col]+1])
                except:  # It is a string
                    csv_data[2*col+1] = rows[pos[col]+1]
            # Make sure the string input is lowercase
            csv_data[0::2] = [x.lower() for x in csv_data[0::2]]
            imported_var.append(csv_data)

    return imported_var
