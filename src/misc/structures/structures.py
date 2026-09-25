"""
Core PET data structures.

This module defines `PETDataFrame`, a pandas `DataFrame` subclass for
ensemble-style tabular data, and `PETStateArray`, a NumPy `ndarray`
subclass for state vectors with PET-specific indexing metadata.
"""

import pandas as pd
import numpy  as np

from pandas._typing import Axes, Dtype

__author__ = 'Mathias Methlie Nilsen'

__all__ = ['PETDataFrame']


class PETDataFrame(pd.DataFrame):
    """
    Pandas DataFrame subclass that preserves all pandas behavior
    while allowing project-specific custom methods.
    """

    # Custom attributes to preserve across pandas operations
    _metadata = [
        'name', 'is_ensemble', 'is_scaled',
        'scale_min', 'scale_max', 'scale_mean', 'scale_std'
    ]

    @property
    def _constructor(self):
        # Ensures pandas ops (copy, loc filtering, arithmetic, etc.)
        # return this subclass when possible.
        return PETDataFrame

    def __init__(
        self,
        data=None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
        name: str | None = None,
        is_ensemble: bool = False,  # Optional flag to indicate if this DataFrame is an ensemble
    ) -> None:

        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.name = name
        self.is_ensemble = is_ensemble
        self.is_scaled = False  # Flag to track if the DataFrame has been scaled

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, name: str | None = None, is_ensemble: bool = False) -> "PETDataFrame":
        """Create a PETDataFrame from an existing pd.DataFrame."""
        out = cls(data=df, name=name, is_ensemble=is_ensemble)
        out.index.name = df.index.name
        out.attrs = df.attrs.copy()
        return out

    @classmethod
    def from_pickle(cls, filepath: str) -> "PETDataFrame":
        """Load a PETDataFrame from a pickle file."""
        df = pd.read_pickle(filepath)
        df.where(pd.notnull(df), None)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Pickle file {filepath} does not contain a DataFrame.")
        return cls.from_pandas(df)

    @classmethod
    def from_csv(cls, filepath: str, **kwargs) -> "PETDataFrame":
        """Load a PETDataFrame from a CSV file."""
        df = pd.read_csv(filepath, **kwargs)
        df.where(pd.notnull(df), None)
        return cls.from_pandas(df)

    @classmethod
    def merge_dataframes(cls, dfs: list[pd.DataFrame]) -> "PETDataFrame":
        '''
        Combine a list of DataFrames (one per ensemble member) into a single
        PETDataFrame where each cell contains an array of ensemble values.
        '''
        if len(dfs) == 0:
            raise ValueError('dfs must contain at least one DataFrame.')
        if not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise ValueError('All elements in dfs must be pandas DataFrames.')

        first = dfs[0]
        for i, dfn in enumerate(dfs[1:], start=1):
            if not dfn.index.equals(first.index):
                raise ValueError(f'DataFrame at position {i} has a different index.')
            if not dfn.columns.equals(first.columns):
                raise ValueError(f'DataFrame at position {i} has different columns.')

        merged = pd.DataFrame(index=first.index, columns=first.columns, dtype=object)
        merged.index.name = first.index.name

        for idx in merged.index:
            for col in merged.columns:
                values = [dfn.at[idx, col] for dfn in dfs]
                merged.at[idx, col] = np.asarray(values).squeeze().T

        out = cls.from_pandas(merged, name=getattr(first, 'name', None), is_ensemble=True)
        out.attrs = first.attrs.copy()
        return out

    def filter_dataframe(self, index=None, columns=None) -> "PETDataFrame":
        """Return a new PETDataFrame filtered to the specified columns and index."""
        filtered = self.copy()
        if index is not None:
            # Let .loc decide whether the labels are present: comparing dtypes
            # rejects indices that select perfectly well (datetime.date labels
            # against a DatetimeIndex, for instance).
            try:
                filtered = filtered.loc[index]
            except KeyError as exc:
                raise ValueError(
                    f"Provided index does not match DataFrame index: {exc}"
                ) from exc
        if columns is not None:
            filtered = filtered.filter(items=columns)

        return filtered


    def scale(self, type='max-min', **kwargs) -> None:
        '''
        Scale each column of DataFrame using the specified method.
        '''
        if type == 'max-min':
            if self.is_scaled:
                raise ValueError("DataFrame is already scaled, cannot apply max-min scaling again without inverting first.")

            self.is_scaled = True
            self.scale_min = self.min() if kwargs.get('minimum', None) is None else kwargs.get('minimum')
            self.scale_max = self.max() if kwargs.get('maximum', None) is None else kwargs.get('maximum')
            scale_range = self.scale_max - self.scale_min

            if isinstance(self.columns, pd.MultiIndex) and (isinstance(self.scale_min, pd.Series) or isinstance(self.scale_max, pd.Series)):
                self.loc[:, :] = self.sub(self.scale_min, axis='columns', level=0).div(scale_range, axis='columns', level=0)
            else:
                self.loc[:, :] = (self - self.scale_min) / scale_range

        elif type == 'z-score':
            if self.is_scaled:
                raise ValueError("DataFrame is already scaled, cannot apply z-score scaling again without inverting first.")
            self.is_scaled = True
            self.scale_mean = self.mean() if kwargs.get('mean', None) is None else kwargs.get('mean')
            self.scale_std = self.std() if kwargs.get('std', None) is None else kwargs.get('std')
            self.loc[:, :] = (self - self.scale_mean) / self.scale_std

        else:
            raise ValueError(f"Unsupported scaling type: {type}")

    def invert_scale(self, type='max-min', **kwargs) -> None:
        '''
        Invert the scaling transformation applied to the DataFrame.
        '''
        if not self.is_scaled:
            raise ValueError("DataFrame is not scaled, cannot invert scale.")
        if type == 'max-min':
            if not self.is_scaled:
                raise ValueError("DataFrame is not scaled, cannot invert max-min scaling.")
            scale_max = self.scale_max if kwargs.get('maximum', None) is None else kwargs.get('maximum')
            scale_min = self.scale_min if kwargs.get('minimum', None) is None else kwargs.get('minimum')
            scale_range = scale_max - scale_min

            if isinstance(self.columns, pd.MultiIndex) and (isinstance(scale_min, pd.Series) or isinstance(scale_max, pd.Series)):
                self.loc[:, :] = self.mul(scale_range, axis='columns', level=0).add(scale_min, axis='columns', level=0)
            else:
                self.loc[:, :] = self * scale_range + scale_min

            self.is_scaled = False

        elif type == 'z-score':
            if not self.is_scaled:
                raise ValueError("DataFrame is not scaled, cannot invert z-score scaling.")
            scale_mean = self.scale_mean if kwargs.get('mean', None) is None else kwargs.get('mean')
            scale_std = self.scale_std if kwargs.get('std', None) is None else kwargs.get('std')
            self.loc[:, :] = self * scale_std + scale_mean
            self.is_scaled = False
        else:
            raise ValueError(f"Unsupported scaling type: {type}")


    def to_series(self) -> pd.Series:
        """Cells as a Series indexed by ``(label, datatype)``, label-major: the legacy flatten order."""
        mult_index = []
        for idx in self.index:
            for col in self.columns:
                mult_index.append((idx, col))
        mult_index = pd.MultiIndex.from_tuples(mult_index, names=[self.index.name, 'datatype'])

        values = []
        for idx in self.index:
            for col in self.columns:
                values.append(self.loc[idx, col])

        return pd.Series(values, index=mult_index)


    def to_matrix(self, filter=True, is_jacobian=False, squeeze=True) -> np.ndarray:
        """Legacy flatten of the observed cells, label-major then type; ``misc.structures.DataLayout`` is the analysis path's equivalent."""

        # If multi-index columns, convert to single-level first
        if isinstance(self.columns, pd.MultiIndex):
            df = self._to_singlelevel_columns()
        else:
            df = self

        arr = []
        for val in df.to_series().values:
            if filter and not np.any(pd.notna(np.atleast_1d(val))):
                continue

            if (not self.is_ensemble) and isinstance(val, np.ndarray) and (not is_jacobian):
                arr.extend(val)
            else:
                arr.append(val)

        if is_jacobian:
            arr = np.stack(arr, axis=0)
        else:
            arr = np.vstack(arr)

        return np.squeeze(arr) if squeeze else arr


    def _to_singlelevel_columns(self) -> "PETDataFrame":
        """
        Convert a MultiIndex-column DataFrame with structure (key, param)
        into a DataFrame with one column per key, where the value is
        the concatenation of all param-arrays for that key.
        """
        result = {}
        keys = pd.Index(self.columns.get_level_values(0)).unique()

        for key in keys:
            param_arrays = self[key]
            concatenated = [
                np.concatenate(param_arrays.iloc[i].values)
                for i in range(len(self))
            ]
            result[key] = concatenated

        df_new = PETDataFrame(result, index=self.index)
        df_new.index.name = self.index.name
        return df_new
