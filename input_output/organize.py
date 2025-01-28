"""Descriptive description."""

from copy import deepcopy
import csv
import datetime as dt


class Organize_input():
    def __init__(self, keys_pr, keys_fwd, keys_en=None):
        self.keys_pr = keys_pr
        self.keys_fwd = keys_fwd
        self.keys_en = keys_en

    def organize(self):
        # Organize the data types given by DATATYPE keyword
        self._org_datatype()
        # Organize the observed data given by TRUEDATA keyword and initialize predicted data variable
        self._org_report()

    def get_keys_pr(self):
        return deepcopy(self.keys_pr)

    def get_keys_fwd(self):
        return deepcopy(self.keys_fwd)

    def get_keys_en(self):
        return deepcopy(self.keys_en)

    def _org_datatype(self):
        """ Check if datatype is given as a csv file. If so, we read and make a list."""
        if isinstance(self.keys_fwd['datatype'], str) and self.keys_fwd['datatype'].endswith('.csv'):
            with open(self.keys_fwd['datatype']) as csvfile:
                reader = csv.reader(csvfile)  # get a reader object
                datatype = []  # Initialize the list of csv data
                for rows in reader:  # Rows is a list of values in the csv file
                    csv_data = [None] * len(rows)
                    for col in range(len(rows)):
                        csv_data[col] = str(rows[col])
                    datatype.extend(csv_data)
            self.keys_fwd['datatype'] = datatype

        if not isinstance(self.keys_fwd['datatype'], list):
            self.keys_fwd['datatype'] = [self.keys_fwd['datatype']]
        # make copy for problem keywords
        self.keys_pr['datatype'] = self.keys_fwd['datatype']

    def _org_report(self):
        """
        Organize the input true observed data. The obs_data will be a list of length equal length of "TRUEDATAINDEX",
        and each entery in the list will be a dictionary with keys equal to the "DATATYPE".
        Also, the pred_data variable (predicted data or forward simulation) will be initialized here with the same
        structure as the obs_data variable.

        !!! warning
            An "N/A" entry in "TRUEDATA" is treated as a None-entry; that is, there is NOT an observed data at this
            assimilation step.'

        !!! warning
            The array associated with the first string inputted in "TRUEDATAINDEX" is assumed to be the "main"
            index, that is, the length of this array will determine the length of the obs_data list! There arrays
            associated with the subsequent strings in "TRUEDATAINDEX" are then assumed to be a subset of the first
            string.
            An example: the first string is SOURCE (e.g., sources in CSEM), where the array will be a list of numbering
            for the sources; and the second string is FREQ, where the array associated will be a list of frequencies.

        !!! info
            It is assumed that the number of data associated with a subset is the same for each index in the subset.
            For example: If two frequencies are inputted in FREQ, then the number of data for one SOURCE index and one
            frequency is 1/2 of the total no. of data for that SOURCE index. If three frequencies are inputted, the number
            of data for one SOURCE index and one frequencies is 1/3 of the total no of data for that SOURCE index,
            and so on.
        """

        # Extract primary indices from "TRUEDATAINDEX"
        if 'truedataindex' in self.keys_pr:

            if isinstance(self.keys_pr['truedataindex'], list):  # List of prim. ind
                true_prim = self.keys_pr['truedataindex']
            else:  # Float
                true_prim = [self.keys_pr['truedataindex']]

            # Check if a csv file has been included as "TRUEDATAINDEX". If so, we read it and make a list,
            if isinstance(self.keys_pr['truedataindex'], str) and self.keys_pr['truedataindex'].endswith('.csv'):
                with open(self.keys_pr['truedataindex']) as csvfile:
                    reader = csv.reader(csvfile)  # get a reader object
                    true_prim = []  # Initialize the list of csv data
                    for rows in reader:  # Rows is a list of values in the csv file
                        csv_data = [None] * len(rows)
                        for ind, col in enumerate(rows):
                            csv_data[ind] = int(col)
                        true_prim.extend(csv_data)
            self.keys_pr['truedataindex'] = true_prim

        # Check if a csv file has been included as "REPORTPOINT". If so, we read it and make a list,
        if 'reportpoint' in self.keys_fwd:
            if isinstance(self.keys_fwd['reportpoint'], str) and self.keys_fwd['reportpoint'].endswith('.csv'):
                with open(self.keys_fwd['reportpoint']) as csvfile:
                    reader = csv.reader(csvfile)  # get a reader object
                    pred_prim = []  # Initialize the list of csv data
                    for rows in reader:  # Rows is a list of values in the csv file
                        csv_data = [None] * len(rows)
                        for ind, col in enumerate(rows):
                            try:
                                csv_data[ind] = int(col)
                            except ValueError:
                                csv_data[ind] = dt.datetime.strptime(
                                    col, '%Y-%m-%d %H:%M:%S')

                        pred_prim.extend(csv_data)
                self.keys_fwd['reportpoint'] = pred_prim

        # Check if assimindex is given as a csv file. If so, we read and make a potential 2D list (if sequential).
        if 'assimindex' in self.keys_pr:
            if isinstance(self.keys_pr['assimindex'], str) and self.keys_pr['assimindex'].endswith('.csv'):
                with open(self.keys_pr['assimindex']) as csvfile:
                    reader = csv.reader(csvfile)  # get a reader object
                    assimindx = []  # Initialize the 2D list of csv data
                    for rows in reader:  # Rows is a list of values in the csv file
                        csv_data = [None] * len(rows)
                        for col in range(len(rows)):
                            csv_data[col] = int(rows[col])
                        assimindx.append(csv_data)
                self.keys_pr['assimindex'] = assimindx

            # check that they are lists
            if not isinstance(self.keys_pr['truedataindex'], list):
                self.keys_pr['truedataindex'] = [self.keys_pr['truedataindex']]
            if not isinstance(self.keys_fwd['reportpoint'], list):
                self.keys_fwd['reportpoint'] = [self.keys_fwd['reportpoint']]
            if not isinstance(self.keys_pr['assimindex'], list):
                self.keys_pr['assimindex'] = [self.keys_pr['assimindex']]
