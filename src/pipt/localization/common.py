"""Localization strategies and shared primitives for PIPT.

Design principles:
- Keep parsing, geometry, and adaptive math in separate classes.
- Expose small, explicit workflow strategies with a stable API.
"""
import csv
import pickle
import numpy as np
from abc import ABC
from typing import Any, Dict, List, Tuple, Union
from pipt.misc_tools.extract_tools import list_to_dict

__all__ = [
    "LocalizationBase",
    "LocalizationConfigBuilder",
    "parse_init_args",
    "normalize_parsed_info",
    "infer_name",
]

class LocalizationBase(ABC):
    """Shared base for localization engines and workflow strategies."""

    def config_common(self, info: Union[dict, list]) -> dict:
        """
        Configure the common localization parameters for all strategies.

        Parameters
        ----------
        info : dict or list
            Localization configuration information.
            - `field`: list of integers specifying the localization field dimensions.
            - `actnum`: optional path to a .npz file containing the actnum array

        """
        if 'field' not in info:
            raise KeyError("'field' must be defined in localization input")
        else:
            assert isinstance(info['field'], list), "'field' must be a list of integers"

        self.info = info

        # Extract and validate the field dimensions
        field = [int(elem) for elem in info['field']]

        # Handle optional actnum file
        actnum = info.get('actnum', None)
        if actnum is not None:
            if not str(actnum).endswith(".npz"):
                raise ValueError("actnum must point to a .npz file")
            actnum_npz = np.load(actnum)
            if hasattr(actnum_npz, "files") and len(actnum_npz.files) > 0:
                key = "actnum" if "actnum" in actnum_npz.files else actnum_npz.files[0]
                actnum = actnum_npz[key]
            else:
                actnum = actnum_npz

        return field, actnum



class LocalizationConfigBuilder:
    """Build normalized localization configuration and precomputed masks."""

    def __init__(self, parsed_info: Union[dict, list]):
        self.parsed_dict = normalize_parsed_info(parsed_info)

    def build(self, data_index: list, data_types: list, parameters: list, ne: int) -> dict:
        """Build the localization info dictionary used by strategies."""
        if "field" not in self.parsed_dict:
            raise KeyError("'field' must be defined in localization input")

        loc_info: Dict[Any, Any] = {
            "field": [int(elem) for elem in self.parsed_dict["field"]],
            "actnum": None,
        }

        if "actnum" in self.parsed_dict and self.parsed_dict["actnum"] is not None:
            file_path = self.parsed_dict["actnum"]
            if not str(file_path).endswith(".npz"):
                raise ValueError("actnum must point to a .npz file")
            actnum_npz = np.load(file_path)
            if hasattr(actnum_npz, "files") and len(actnum_npz.files) > 0:
                key = "actnum" if "actnum" in actnum_npz.files else actnum_npz.files[0]
                loc_info["actnum"] = actnum_npz[key]
            else:
                loc_info["actnum"] = actnum_npz

        if "threshold" in self.parsed_dict:
            loc_info["threshold"] = self.parsed_dict["threshold"]

        mode_info = self._parse_special_modes(self.parsed_dict)
        if mode_info is not None:
            loc_info.update(mode_info)
            loc_info["mask"] = {}
            return loc_info
        else:
            pickle_data = self._load_pickle_localization(self.parsed_dict)
            if pickle_data is not None:
                loc_info = pickle_data
                if "field" not in loc_info:
                    loc_info["field"] = [int(elem) for elem in self.parsed_dict["field"]]
                if "actnum" not in loc_info:
                    loc_info["actnum"] = None
                if "threshold" in self.parsed_dict and "threshold" not in loc_info:
                    loc_info["threshold"] = self.parsed_dict["threshold"]
            else:
                loc_info = self._build_explicit_localization_entries(
                    parsed_dict=self.parsed_dict,
                    data_index=data_index,
                    data_types=data_types,
                    parameters=parameters,
                    init_local=loc_info,
                )

        # NOTE: SpatialLocalization (from distance_loc) removed — LocalAnalysisLocalization
        # will be reimplemented without LocalizationConfigBuilder.
        loc_info["mask"] = {}
        return loc_info

    @staticmethod
    def _parse_special_modes(parsed_dict: dict) -> Union[dict, None]:
        if "autoadaloc" in parsed_dict:
            mode = {
                "autoadaloc": True,
                "nstd": parsed_dict["autoadaloc"],
            }
            if "type" in parsed_dict:
                mode["type"] = parsed_dict["type"]
            return mode

        if "localanalysis" in parsed_dict:
            mode = {"localanalysis": True}
            if "type" in parsed_dict:
                mode["type"] = parsed_dict["type"]
            if "range" in parsed_dict:
                mode["range"] = float(parsed_dict["range"])
            return mode

        return None

    @staticmethod
    def _load_pickle_localization(parsed_dict: dict) -> Union[dict, None]:
        pickle_file = None
        for _, value in parsed_dict.items():
            if str(value).endswith(".p") or str(value).endswith(".pkl"):
                pickle_file = value
                break
        if pickle_file is None:
            return None
        with open(pickle_file, "rb") as stream:
            return pickle.load(stream)

    def _build_explicit_localization_entries(
        self,
        parsed_dict: dict,
        data_index: list,
        data_types: list,
        parameters: list,
        init_local: dict,
    ) -> dict:
        for time in data_index:
            for datum in data_types:
                for parameter in parameters:
                    init_local[(datum, time, parameter)] = {
                        "taper_func": None,
                        "position": None,
                        "anisotropi": None,
                        "range": None,
                    }

        info_rows = self._read_localization_rows(parsed_dict)
        for row in info_rows:
            self._apply_localization_row(row, init_local)

        return init_local

    @staticmethod
    def _read_localization_rows(parsed_dict: dict) -> List[str]:
        csv_key = next((k for k in parsed_dict if str(k).endswith(".csv")), None)
        if csv_key:
            with open(csv_key) as csv_file:
                reader = csv.reader(csv_file)
                return [item for sublist in reader for item in sublist]

        for key in parsed_dict:
            if len(str(key).split(",")) > 1:
                return str(key).split(",")

        return []

    @staticmethod
    def _apply_localization_row(row: str, init_local: dict) -> None:
        tmp_info = row.split()
        if not tmp_info:
            return

        if len(tmp_info) == 11:
            name = (tmp_info[8].lower(), float(tmp_info[9]), tmp_info[10].lower())
        else:
            name = (
                tmp_info[8].lower() + " " + tmp_info[9].lower(),
                float(tmp_info[10]),
                tmp_info[11].lower(),
            )

        if name not in init_local:
            return

        entry = init_local[name]
        entry["taper_func"] = tmp_info[0]

        if tmp_info[0] == "import":
            entry["file"] = tmp_info[1]
            return

        entry["position"] = [[int(float(tmp_info[1])), int(float(tmp_info[2])), int(float(tmp_info[3]))]]
        entry["range"] = [int(tmp_info[4]), int(tmp_info[5])]
        entry["anisotropi"] = [float(tmp_info[6]), float(tmp_info[7])]

    def _build_unique_masks(self, init_local: dict, ne: int, spatial_engine) -> dict:
        masks: Dict[Any, np.ndarray] = {}

        loc_mask_info = [
            (
                init_local[element]["taper_func"],
                init_local[element]["anisotropi"][0],
                init_local[element]["anisotropi"][1],
                init_local[element]["range"],
            )
            for element in init_local.keys()
            if isinstance(element, tuple) and len(element) == 3 and init_local[element]["taper_func"] is not None
        ]

        for info in loc_mask_info:
            key, loc_range = self._mask_key_from_info(info)
            if key in masks:
                continue

            masks[key] = spatial_engine.gen_loc_mask(
                taper_function=info[0],
                anisotropi=[info[1], info[2]],
                loc_range=loc_range,
                field_size=init_local["field"],
                ne=ne,
            )

        return masks

    @staticmethod
    def _mask_key_from_info(info: tuple) -> Tuple[tuple, Any]:
        taper_func, aniso_1, aniso_2, loc_range = info

        if taper_func == "region":
            if isinstance(loc_range, list):
                return ("region", loc_range[0], loc_range[1], loc_range[2]), loc_range
            return ("region", loc_range), loc_range

        if isinstance(loc_range, list):
            return (taper_func, aniso_1, aniso_2, loc_range[0], loc_range[1]), loc_range[0]

        return (taper_func, aniso_1, aniso_2, loc_range), loc_range


#: The keyword whose *presence* selected each mode before the strategies were named.
#: Order matters: it is the order the original chain tested them in.
_MODE_KEYWORDS = (
    ("autoadaloc", "autoadaloc"),
    ("localanalysis", "localanalysis"),
    ("dist_loc", "distance_loc"),
)


def infer_name(info: dict) -> str:
    """Name the localization mode a config selects by keyword rather than by name.

    Localization used to be chosen by which keyword appeared in the block --
    ``autoadaloc``, ``localanalysis``, ``dist_loc``, a pickled mask file, or none of
    them for the parallel update. Those configs carry no ``name``, so it is worked out
    here and they keep running unchanged.
    """
    for keyword, name in _MODE_KEYWORDS:
        if keyword in info:
            return name

    # ``dist_loc`` was also accepted as a bare value rather than a key.
    values = [str(value) for value in info.values()]
    if "dist_loc" in values:
        return "distance_loc"

    # A pickled mask file, under any key, means distance localization.
    if any(value.endswith((".p", ".pkl")) for value in values):
        return "distance_loc"

    return "parallel_update"


def normalize_parsed_info(parsed_info: Union[dict, list]) -> dict:
    """Normalize localization input to dictionary form, naming the mode if it does not."""
    if isinstance(parsed_info, list):
        parsed_info = list_to_dict(parsed_info)
    if not isinstance(parsed_info, dict):
        raise TypeError("parsed_info must be dict or list")
    if "name" not in parsed_info:
        parsed_info = {**parsed_info, "name": infer_name(parsed_info)}
    return parsed_info


def parse_init_args(
    data_indices: Union[list, None] = None,
    data_types: Union[list, None] = None,
    parameters: Union[list, None] = None,
    ensemble_size: Union[int, None] = None,
):
    """Parse constructor arguments using canonical keyword names."""

    if data_indices is None or data_types is None or parameters is None or ensemble_size is None:
        raise TypeError(
            "Localization requires data_indices, data_types, parameters, and ensemble_size."
        )

    return data_indices, data_types, parameters, ensemble_size



