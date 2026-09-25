"""Observed data and its variance, read from the files a config names.

:class:`DataReader` turns the ``truedata``/``datavar`` entries of the
``[dataassim]`` block -- CSV, pickle or ``.npz`` files, with cells that may
themselves point at ``.npz`` arrays -- into the frames the ensemble holds,
applying wavelet compression to seismic vintages when configured.
"""

import ast
import os
from copy import deepcopy

import numpy as np

from misc.structures import PETDataFrame

class DataReader:
    """Reads the observed data and its variance, as frames, from the files a config names."""

    def __init__(self, info: dict, **kwargs):
        self.info = info
        self.data = info.get('data', None)
        self.var = info.get('datavar', None)

        # NB: Not sure if this will be used or needed!
        self.assimindex  = info.get('assimindex', None)
        self.truedataindex = info.get('truedataindex', None)
        self.datatype = info.get('datatype', None)

        # Sparse infor for data compression (for seismic data)
        self.sparse = kwargs.get('sparse_info', None)
        self.sparse_data = []

        # Error handling for missing data or variance
        if self.data is None:
            msg = "Data missing: the 'data' key is missing in info dictionary."
            raise ValueError(msg)
        if self.var is None:
            msg = "Variance missing: the 'datavar' key is missing in info dictionary."
            raise ValueError(msg)


    def get_data(self) -> PETDataFrame:
        """The observations as a frame: report labels as index, data types as columns; ``.npz`` cells are loaded and compressed vintages reduced to their leading wavelet coefficients."""
        if isinstance(self.data, str):
            df = self._read_from_file(self.data)
        elif isinstance(self.data, dict):
            df = self._read_from_dict(self.data)
        else:
            msg = f"Unsupported data type: {type(self.data)}. Expected str or dict."
            raise TypeError(msg)

        # Process each cell for potential npz files and apply wavelet compression if specified
        vintage = 0
        for i, idx in enumerate(df.index):
            for col in df.columns:
                cell = df.loc[idx, col]

                if isinstance(cell, str) and cell.endswith('.npz'):
                    npzfile = np.load(cell, allow_pickle=True)
                    cell = np.squeeze(npzfile[npzfile.files[0]])
                    assert cell.ndim < 2, f"Expected 1D array in npz file {cell}, but got {cell.ndim}D."

                if (self.sparse is not None) and (col in self.sparse['compress_data']) and (not np.isnan(cell).any()):
                    if vintage < len(self.sparse['mask']):
                        cell = self._wavelet_compression(cell, vintage=vintage)
                        vintage += 1

                # Store new value
                df.at[idx, col] = cell

        # NB: Not sure if this will be used or needed!
        self.datatype = df.columns.tolist()
        self.assimindex = np.arange(len(df.index)).tolist()
        self.truedataindex = df.index.tolist()
        return df


    def get_variance(self, data_df: PETDataFrame, sparse_data: list=None) -> PETDataFrame:
        """The variance frame on ``data_df``'s geometry, from ``['abs', v]``, ``['rel', percent]``, ``['emp', ensemble]`` or ``['cd', file]`` cells; a compressed vintage gets its estimated noise squared."""
        if isinstance(self.var, str):
            _df = self._read_from_file(self.var)
        else:
            msg = f"Unsupported variance type: {type(self.var)}. Expected str (file path)."
            raise TypeError(msg)

        # Fill in dataframe
        vintage = 0
        df = PETDataFrame(columns=data_df.columns, index=data_df.index)
        for i, idx in enumerate(data_df.index):
            for c, col in enumerate(data_df.columns):

                if (data_df.loc[idx, col] is not None) and (not np.isnan(data_df.loc[idx, col]).any()):
                    # Sparse stuff (for seismic data)
                    if (
                        self.sparse is not None
                        and sparse_data is not None
                        and col in self.sparse.get('compress_data', [])
                        and vintage < len(sparse_data)
                    ):
                        var = np.power(sparse_data[vintage].est_noise, 2)
                        vintage += 1


                    else:
                        var = self._extract_cell_variance(
                            _df.loc[idx, col],
                            data_df.loc[idx, col],
                            i,
                            c,
                        )

                    df.at[idx, col] = var
                else:
                    df.at[idx, col] = None

        # Mark as ensemble if specified in info
        if 'emp_cov' in self.info:
            if (self.info['emp_cov'] == 'yes') or (self.info['emp_cov'] is True):
                df.is_ensemble = True

        return df.astype(float, errors='ignore')


    def _read_from_file(self, filepath: str) -> PETDataFrame:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pkl':
            df = PETDataFrame.from_pickle(filepath)
        elif ext == '.csv':
            df = PETDataFrame.from_csv(filepath, index_col=0, parse_dates=True)
            #df = df.astype(float, errors='ignore')
        elif ext == '.npz':
            data = dict(np.load(filepath, allow_pickle=True))
            df = self._read_from_dict(data)
        else:
            msg = f"Unsupported file type: {filepath}. Expected .csv, .pkl, or .npz."
            raise ValueError(msg)
        return df


    def _read_from_dict(self, data_dict: dict) -> PETDataFrame:
        index = data_dict.pop('index', None)
        index_name = data_dict.pop('index_name', self.info.get('obsname', None))
        df = PETDataFrame(data=data_dict, index=index)
        df.index.name = index_name
        return df


    def _extract_cell_variance(self, var_cell, data_cell, i, c):

        if isinstance(var_cell, str) and var_cell.strip().startswith('['):
            # Example: "['abs', 0.5]" --> ['abs', 0.5]
            var_cell = ast.literal_eval(var_cell)

        # Variance given as relative percentage (e.g., ['rel', 5] means 5% of the data value)
        if var_cell[0].lower() == 'rel':
            if var_cell[1] is None:
                return None
            return (0.01*var_cell[1] * data_cell)**2

        # Variance given as absolute value (e.g., ['abs', 0.5] means a variance of 0.5).
        # If the value is iterable, it is indexed by column.
        elif var_cell[0].lower() == 'abs':
            if hasattr(data_cell, 'ndim') and data_cell.ndim > 0:
                val = var_cell[1]*np.ones_like(data_cell)
                return val
            else:
                val = var_cell[1]
            if hasattr(val, '__iter__') and not isinstance(val, str):
                return val[c]
            else:
                return val

        # Variance given as empirical ensemble (e.g., ['emp', [300, 350, 244, ...]]).
        elif (var_cell[0].lower() == 'emp'):
            return var_cell[1]

        # Variance given as full covariance matrix (e.g., ['cd', 'covfile.npz']).
        elif (var_cell[0].lower() == 'cd') and (var_cell[1].endswith('.npz')):
            # Populate once
            if not hasattr(self, 'cov'):
                covfile = np.load(var_cell[1], allow_pickle=True)['cov']
                self.cov = covfile[covfile.files[0]]
            return self.cov[i*c, i*c]

        # Return None if no data for this cell
        elif data_cell is None:
            return None

        else:
            msg = f"Unsupported variance type in cell: {var_cell}. Expected format like ['rel', value], ['abs', values], or ['emp', value]."
            raise ValueError(msg)


    def _wavelet_compression(self, arr, vintage):

        options = deepcopy(self.sparse)
        options['mask'] = options['mask'][vintage]
        min_noise = options['min_noise']

        if isinstance(min_noise, list):
            if 0 <= vintage < len(min_noise):
                options['min_noise'] = min_noise[vintage]
            else:
                msg = 'min_noise must either be scalar or list with one number for each vintage'
                raise ValueError(msg)

        # Apply wavelet compression. Imported here: PyWavelets is needed only
        # for sparse compression, and keeping the import out of module scope
        # means importing misc does not import pipt.
        from pipt.misc_tools.wavelet_tools import SparseRepresentation

        sparsrep = SparseRepresentation(options)
        arr_compressed, wdec_rec = sparsrep.compress(arr, th_mult=options['th_mult'])
        self.sparse_data.append(sparsrep) # Store the information

        # Save reconstructed data
        arr_reconstructed = sparsrep.reconstruct(wdec_rec)  # reconstruct the data
        np.savez('truedata_rec_' + str(vintage) + '.npz', arr_reconstructed)

        return arr_compressed

