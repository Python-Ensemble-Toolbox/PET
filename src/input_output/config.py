"""The configuration boundary: one normalisation and one validation, however a config arrives.

A run is described by three sections -- the problem (``dataassim`` for pipt,
``optim`` for popt), the ``ensemble`` and the ``simulator`` (``fwdsim``) --
read from TOML, YAML or the legacy ``.pipt``/``.popt`` text format, or built
as dictionaries in a script. Whichever way they arrive, :func:`normalize`
turns them into the one form the rest of PET reads: the canonical name where
a key has had several spellings, a boolean where a flag could be ``yes``/``no``,
a dictionary where a sub-block could be a list of pairs, and the field
conversions (``datatype``, ``reportpoint``, ``assimindex``) done once. The
result is a copy; nothing downstream sees, or changes, the caller's
dictionaries. :func:`validate` says what a run would fail on, by section and
key, instead of an assertion or a ``KeyError`` somewhere inside a scheme.
"""

from copy import deepcopy
from dataclasses import dataclass

from input_output.organize import ConfigNormalizer

__all__ = ["ConfigError", "Problem", "as_flag", "pairs_to_dict", "normalize", "normalize_dataassim",
           "normalize_ensemble", "normalize_simulator", "normalize_optim", "validate", "fatal_problems",
           "KNOWN_DATAASSIM", "KNOWN_ENSEMBLE"]


class ConfigError(ValueError):
    """A config that cannot run, with every problem listed."""


# ---------------------------------------------------------------------------
# Value helpers (the legacy text format wrote flags as yes/no and blocks as rows)
# ---------------------------------------------------------------------------
def as_flag(value, default=False) -> bool:
    """A boolean from a flag value: booleans as they are, ``yes``/``no``/``true``/``false`` strings, else truthiness."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("yes", "true"):
            return True
        if lowered in ("no", "false"):
            return False
    return bool(value)


def pairs_to_dict(entries) -> dict:
    """``[[key, value], [key], [key, v1, v2]]`` -> ``{key: value, key: None, key: [v1, v2]}``."""
    assert isinstance(entries, list)
    result = {}
    for entry in entries:
        if not isinstance(entry, list):
            entry = [entry]
        if len(entry) == 1:
            result[str(entry[0])] = None
        elif len(entry) == 2:
            result[str(entry[0])] = entry[1]
        else:
            result[str(entry[0])] = entry[1:]
    return result


# ---------------------------------------------------------------------------
# What each section canonicalises
# ---------------------------------------------------------------------------
ALIASES_DATAASSIM = {"truedata": "data", "var": "datavar", "save_folder": "savefolder", "restartfile": "restart_file"}
FLAGS_DATAASSIM = ("emp_cov", "restart", "restartsave", "obsvarsave", "screendata", "post_process_forecast",
                   "scale_data", "logit")
BLOCKS_DATAASSIM = ("iteration", "mda", "compress", "localization", "localanalysis")

ALIASES_ENSEMBLE = {"importstaticvar": "importstate", "save_folder": "savefolder"}
FLAGS_ENSEMBLE = ("save_prior", "disable_tqdm", "natural_gradient")
BLOCKS_ENSEMBLE = ("multilevel",)

ALIASES_SIMULATOR: dict = {}
FLAGS_SIMULATOR = ("compute_adjoints", "replace", "hpc")
BLOCKS_SIMULATOR: tuple = ()

ALIASES_OPTIM = {"save_folder": "savefolder", "restartfile": "restart_file"}
FLAGS_OPTIM = ("restart", "restartsave", "saveit", "logit", "transform")
BLOCKS_OPTIM: tuple = ()

#: Keys the code reads from the two sections PET owns. `pet validate` points out anything else,
#: since a misspelt key is silently ignored otherwise. Simulator keys are the wrapper's business.
KNOWN_DATAASSIM = frozenset({
    "scheme", "analysis", "data", "datavar", "obsname", "datatype", "truedataindex", "assimindex", "energy",
    "emp_cov", "iteration", "mda", "compress", "localization", "localanalysis", "actnum", "scale_data", "scale",
    "screendata", "post_process_forecast", "remove_outliers", "add_synthetic_noise",
    "savefolder", "nosave", "savedata", "analysisdebug", "iterinfo", "obsvarsave", "qa", "qc",
    "restart", "restartsave", "restart_file", "logit", "logger_name",
    # legacy text files keep the ensemble's keys in DATAASSIM
    "ne", "state", "staticvar", "importstate", "seed", "save_prior", "sim_limit", "disable_tqdm",
})
KNOWN_ENSEMBLE = frozenset({
    "ne", "state", "controls", "importstate", "seed", "save_prior", "sim_limit", "disable_tqdm", "multilevel",
    "savefolder", "natural_gradient", "num_models", "save_prediction",
})


def _canonical(keys, aliases, flags, blocks, block_prefixes=()):
    keys = deepcopy(keys) if keys else {}
    for old, new in aliases.items():
        if old in keys:
            keys.setdefault(new, keys[old])   # the canonical spelling wins when both are given
            del keys[old]
    for key in flags:
        if key in keys:
            keys[key] = as_flag(keys[key])
    for key in list(keys):
        if (key in blocks or key.startswith(block_prefixes)) and isinstance(keys[key], list):
            keys[key] = pairs_to_dict(keys[key])
    return keys


def normalize_dataassim(keys) -> dict:
    """The ``dataassim`` section in canonical form, as a copy."""
    return _canonical(keys, ALIASES_DATAASSIM, FLAGS_DATAASSIM, BLOCKS_DATAASSIM, block_prefixes=("prior_",))


def normalize_ensemble(keys) -> dict:
    """The ``ensemble`` section in canonical form, as a copy."""
    return _canonical(keys, ALIASES_ENSEMBLE, FLAGS_ENSEMBLE, BLOCKS_ENSEMBLE, block_prefixes=("prior_",))


def normalize_simulator(keys) -> dict:
    """The ``simulator`` section in canonical form, as a copy."""
    return _canonical(keys, ALIASES_SIMULATOR, FLAGS_SIMULATOR, BLOCKS_SIMULATOR)


def normalize_optim(keys) -> dict:
    """The ``optim`` section in canonical form, as a copy."""
    return _canonical(keys, ALIASES_OPTIM, FLAGS_OPTIM, BLOCKS_OPTIM)


def is_dataassim(problem_section) -> bool:
    """Whether the problem section describes a data-assimilation run (else an optimisation)."""
    return bool(problem_section) and ("scheme" in problem_section or "daalg" in problem_section)


def normalize(cfg_prb, cfg_sim, cfg_ens=None):
    """All three sections as the rest of PET reads them: field conversions, canonical names, flags, blocks.

    Returns ``(problem, simulator, ensemble)``; the ensemble is ``{}`` when the
    config has none (the legacy text format keeps those keys in DATAASSIM).
    """
    cfg_prb, cfg_sim, cfg_ens = ConfigNormalizer.normalize_config(cfg_prb, cfg_sim, cfg_ens)
    problem = normalize_dataassim(cfg_prb) if is_dataassim(cfg_prb) else normalize_optim(cfg_prb)
    return problem, normalize_simulator(cfg_sim), normalize_ensemble(cfg_ens or {})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    """One thing wrong with a config. ``fatal`` problems stop a run at construction."""

    section: str
    key: str
    message: str
    fatal: bool = True

    def __str__(self) -> str:
        return f"[{self.section}] {self.key}: {self.message}"


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else [value]


def validate(cfg_prb, cfg_sim=None, cfg_ens=None) -> list:
    """Everything a run would fail on, by section and key; empty when the config is fine.

    Expects normalised sections (see :func:`normalize`). The ensemble's
    requirements are checked when an ensemble section is given or the problem
    section carries its keys, as legacy text files do.
    """
    prb, sim, ens = (cfg_prb or {}), (cfg_sim or {}), (cfg_ens or {})
    problems = []
    if "daalg" in prb:
        problems.append(Problem("dataassim", "daalg", "replaced by `scheme`; run `pet migrate` on the file"))
    if is_dataassim(prb):
        for key, what in (("data", "the observed data"), ("datavar", "the observation variance")):
            if key not in prb:
                problems.append(Problem("dataassim", key, f"required: {what}"))
        data = prb.get("data")
        if "obsname" not in prb and not (isinstance(data, dict) and "index_name" in data):
            problems.append(Problem("dataassim", "obsname", "required: the name of the observation index (times, dates)"))
    if sim and "datatype" not in sim and "datatype" not in prb:
        problems.append(Problem("simulator", "datatype", "required: the data types the simulator reports", fatal=False))

    merged = {**prb, **ens}
    if ens or any(key in prb for key in ("ne", "state", "staticvar")):
        if "ne" not in merged:
            problems.append(Problem("ensemble", "ne", "required: the ensemble size", fatal=False))
        state = merged.get("state", merged.get("staticvar"))
        if state is None and "controls" not in merged:
            problems.append(Problem("ensemble", "state", "required: the state variables (or `controls` for optimisation)"))
        elif state is not None and "importstate" not in merged:
            for name in _as_list(state):
                if f"prior_{name}" not in merged:
                    problems.append(Problem("ensemble", f"prior_{name}",
                                            f"required: the prior description of `{name}` (or `importstate` to load one)"))
    return problems


def fatal_problems(cfg_prb, cfg_sim=None, cfg_ens=None) -> list:
    return [problem for problem in validate(cfg_prb, cfg_sim, cfg_ens) if problem.fatal]


def unknown_keys(cfg_prb, cfg_ens=None) -> list:
    """Keys in the two sections PET owns that nothing reads -- usually a misspelling."""
    prb, ens = (cfg_prb or {}), (cfg_ens or {})
    found = []
    if is_dataassim(prb):
        found += [f"[dataassim] {key}" for key in prb if key not in KNOWN_DATAASSIM and not key.startswith("prior_")]
    found += [f"[ensemble] {key}" for key in ens if key not in KNOWN_ENSEMBLE and not key.startswith("prior_")]
    return found
