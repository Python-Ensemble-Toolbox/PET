"""Descriptive description."""

from copy import deepcopy
from pathlib import Path
import csv
import os
import pandas as pd
import yaml



class ConfigNormalizer:
    """
    Utility class for normalizing and type-converting configuration dictionaries for PIPT/POPT workflows.

    This class provides static methods to process and normalize configuration sections such as 'datatype',
    'truedataindex', 'reportpoint', and 'assimindex'.
    """

    @staticmethod
    def normalize_datatype(datatype):
        """
        Normalize the 'datatype' field: read from CSV if needed, ensure list of strings.
        """
        if isinstance(datatype, str) and datatype.endswith('.csv'):
            with open(datatype) as csvfile:
                reader = csv.reader(csvfile)
                return [str(col) for row in reader for col in row]
        if not isinstance(datatype, list):
            return [datatype]
        return [str(x) for x in datatype]

    @staticmethod
    def normalize_truedataindex(truedataindex):
        """
        Normalize the 'truedataindex' field: read from CSV if needed, ensure list of ints.
        """
        if isinstance(truedataindex, str) and truedataindex.endswith('.csv'):
            with open(truedataindex) as csvfile:
                reader = csv.reader(csvfile)
                return [int(col) for row in reader for col in row]
        if not isinstance(truedataindex, list):
            return [truedataindex]
        return [int(x) for x in truedataindex]

    @staticmethod
    def normalize_reportpoint(reportpoint):
        """
        Normalize the 'reportpoint' field: handle CSV, dict (date_range), or pass through.
        """
        if isinstance(reportpoint, str):
            return report_point_file_reader(reportpoint)
        elif isinstance(reportpoint, dict):
            return pd.date_range(**reportpoint).to_pydatetime().tolist()
        elif not isinstance(reportpoint, list):
            return [reportpoint]
        return reportpoint

    @staticmethod
    def normalize_assimindex(assimindex):
        """
        Normalize the 'assimindex' field: read from CSV if needed, ensure list of lists of ints.
        """
        if isinstance(assimindex, str) and assimindex.endswith('.csv'):
            with open(assimindex) as csvfile:
                reader = csv.reader(csvfile)
                return [[int(col) for col in row] for row in reader]
        if not isinstance(assimindex, list):
            return [assimindex]
        # If it's a flat list, wrap in another list
        if assimindex and not isinstance(assimindex[0], list):
            return [assimindex]
        return assimindex

    @staticmethod
    def normalize_config(keys_pr, keys_fwd, keys_en=None):
        """
        Normalize all relevant fields in the config dictionaries and return new dicts.
        """
        keys_pr = deepcopy(keys_pr) if keys_pr else {}
        keys_fwd = deepcopy(keys_fwd) if keys_fwd else {}
        keys_en = deepcopy(keys_en) if keys_en else {} if keys_en is not None else None

        # Normalize datatype
        if 'datatype' in keys_fwd:
            keys_fwd['datatype'] = ConfigNormalizer.normalize_datatype(keys_fwd['datatype'])
            keys_pr['datatype'] = keys_fwd['datatype']

        # Normalize truedataindex
        if 'truedataindex' in keys_pr:
            keys_pr['truedataindex'] = ConfigNormalizer.normalize_truedataindex(keys_pr['truedataindex'])

        # Normalize reportpoint
        if 'reportpoint' in keys_fwd:
            keys_fwd['reportpoint'] = ConfigNormalizer.normalize_reportpoint(keys_fwd['reportpoint'])

        # Normalize assimindex
        if 'assimindex' in keys_pr:
            keys_pr['assimindex'] = ConfigNormalizer.normalize_assimindex(keys_pr['assimindex'])

        return keys_pr, keys_fwd, keys_en


def report_point_file_reader(filepath):
    """
    Read a file containing report points and return parsed values.

    Supported file types:
        - .csv : Each cell is parsed as int or datetime
        - .txt : Each line is parsed as int or datetime
        - .yaml: Each entry is parsed as int or datetime

    Parameters
    ----------
    filepath : str
        Path to the input file.

    Returns
    -------
    list
        List of parsed values (int or datetime-like objects).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is unsupported or parsing fails.
    """

    def _parse_value(value, source):
        """Parse a single value into int or datetime."""
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            return None

        try:
            return int(value)
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(value)
            except Exception:
                raise ValueError(
                    f"Unable to parse '{value}' in file '{source}' "
                    "as integer or datetime."
                )

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    extension = Path(filepath).suffix.lower()
    report_points = []

    if extension == ".csv":
        df = pd.read_csv(filepath, header=None)
        values = df.values.ravel()

        for value in values:
            parsed = _parse_value(value, filepath)
            if parsed is not None:
                report_points.append(parsed)

    elif extension == ".txt":
        with open(filepath, encoding="utf-8") as file:
            for line in file:
                parsed = _parse_value(line.strip(), filepath)
                if parsed is not None:
                    report_points.append(parsed)

    elif extension == ".yaml":
        with open(filepath, encoding="utf-8") as file:
            data = yaml.safe_load(file) or []

        for value in data:
            parsed = _parse_value(value, filepath)
            if parsed is not None:
                report_points.append(parsed)

    else:
        raise ValueError(f"Unsupported file type: '{extension}'")

    return report_points
