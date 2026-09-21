"""Inversion (estimation, data assimilation)

--8<-- "pipt/README.md"
"""

__pdoc__ = {}

# import replacement module and override submodule in pipt
# import sys
# from #replacement_package import #replacement_submodule
# sys.modules["pipt.#module.#submodule"] = #replacement_submodule

from pipt.update_schemes.factory import build_scheme  # noqa: E402
from pipt.update_schemes.enkf import EnKF  # noqa: E402
from pipt.update_schemes.enrml import GNEnRML, LMEnRML  # noqa: E402
from pipt.update_schemes.es import ES  # noqa: E402
from pipt.update_schemes.esmda import ESMDA  # noqa: E402
from pipt.update_schemes.registry import (  # noqa: E402
    available_schemes,
    get_scheme,
    register_scheme,
)

__all__ = [
    "EnKF",
    "ES",
    "ESMDA",
    "LMEnRML",
    "GNEnRML",
    "build_scheme",
    "available_schemes",
    "get_scheme",
    "register_scheme",
]
