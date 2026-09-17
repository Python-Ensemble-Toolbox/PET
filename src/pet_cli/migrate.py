"""Migrate legacy PET config files to the current schema.

Two changes are handled.

The two-element ``daalg`` key, which packed an assimilation family and an
update method into a list, is replaced by a single ``scheme`` key naming the
algorithm::

    daalg = ["esmda", "esmda"]   ->   scheme = "esmda"

The second element was the one that actually selected the class, so that is
what carries over. Where the two elements disagree the second still wins, and
the migration reports it so the change is visible rather than silent.

``analysisdebug`` is renamed to ``savedata``, matching popt and describing
what the key does -- it names the variables recorded each iteration, which is
a record of the run rather than a debugging aid::

    analysisdebug = [...]   ->   savedata = [...]

The old spelling still works at runtime, with a deprecation warning, so this
one is a tidy-up rather than a required migration. The *output files* did
change name, from ``debug_analysis_step_{i}.npz`` to
``assimilation_result_{i}.npz``, which no config rewrite can paper over: any
post-processing that globs the old pattern needs updating by hand.

Formatting is preserved. The rewrite is a surgical edit of the ``daalg``
assignment itself, not a parse-and-redump of the file, because a round trip
through a TOML/YAML writer discards everything that is not data: comments,
commented-out alternative blocks, indentation, inline tables, quote style and
list layout. Real PET configs carry all of those -- a commented-out
localization block that gets toggled against the active one is a common
pattern, and silently deleting it would be unacceptable.

If the surgical edit cannot find the assignment (an unusual layout), the
migration falls back to the round trip and warns that formatting will be lost,
rather than failing or destroying the file silently.
"""

from __future__ import annotations

import re

import shutil
from pathlib import Path

import tomli
import tomli_w
import yaml

__all__ = ["migrate_config", "migrate_section", "MigrationReport"]

_DA_SECTIONS = ("dataassim", "optim")

#: Keys renamed with their value untouched, ``old -> new``.
_RENAMED_KEYS = {"analysisdebug": "savedata"}


class MigrationReport:
    """What a migration changed, or would change."""

    def __init__(self) -> None:
        self.changes: list[str] = []
        self.warnings: list[str] = []

    @property
    def changed(self) -> bool:
        """Whether the migration changed anything."""
        return bool(self.changes)

    def __str__(self) -> str:
        lines = [f"  - {c}" for c in self.changes]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def migrate_section(section: dict, report: MigrationReport) -> dict:
    """Migrate one config section in place, recording what changed."""
    for old, new in _RENAMED_KEYS.items():
        if old not in section:
            continue
        if new in section:
            report.warnings.append(
                f"Section already has '{new}'; left the deprecated '{old}' in "
                f"place rather than guessing which one you meant."
            )
            continue
        section[new] = section.pop(old)
        report.changes.append(f"{old}  ->  {new}")

    if "daalg" not in section:
        return section

    daalg = section.pop("daalg")

    if isinstance(daalg, str):
        scheme = daalg
    elif isinstance(daalg, (list, tuple)) and daalg:
        scheme = daalg[-1]
        if len(daalg) == 2 and daalg[0] != daalg[1]:
            report.warnings.append(
                f"daalg was {list(daalg)!r} with differing entries; "
                f"kept {scheme!r}, which is the one that selected the class."
            )
        elif len(daalg) > 2:
            report.warnings.append(
                f"daalg had {len(daalg)} entries {list(daalg)!r}; kept {scheme!r}."
            )
    else:
        report.warnings.append(
            f"daalg had unexpected value {daalg!r}; left the file unchanged."
        )
        section["daalg"] = daalg
        return section

    section["scheme"] = scheme
    report.changes.append(f"daalg = {daalg!r}  ->  scheme = {scheme!r}")

    if "analysis" not in section:
        report.warnings.append(
            "No 'analysis' key found; add one to select the analysis flavour "
            "(e.g. analysis = 'approx')."
        )

    return section


#: ``daalg`` assignment in TOML: `daalg = [...]`, `daalg = "x"`, `daalg = 'x'`.
#: The list alternative is non-greedy so it also matches a multi-line list.
_TOML_DAALG = re.compile(
    r"^(?P<indent>[^\S\n]*)daalg(?P<pre>[^\S\n]*)=(?P<post>[^\S\n]*)"
    r"(?P<value>\[[\s\S]*?\]|\"[^\"\n]*\"|'[^'\n]*')"
    r"(?P<trail>[^\n]*)$",
    re.MULTILINE,
)

#: Same for YAML inline form: `daalg: [a, b]` or `daalg: x`.
_YAML_DAALG = re.compile(
    r"^(?P<indent>[^\S\n]*)daalg(?P<pre>[^\S\n]*):(?P<post>[^\S\n]*)"
    r"(?P<value>\[[^\]\n]*\]|[^\n#]+?)"
    r"(?P<trail>[^\n]*)$",
    re.MULTILINE,
)


def _replace_daalg_in_text(text: str, fmt: str, scheme: str):
    """Replace the ``daalg`` assignment in place, leaving the rest untouched.

    Returns ``(new_text, count)``. A count of 0 means the assignment could not
    be located and the caller should fall back to a full rewrite.
    """
    pattern = _TOML_DAALG if fmt == "toml" else _YAML_DAALG
    separator = "=" if fmt == "toml" else ":"

    def substitute(match):
        # "scheme" is one character longer than "daalg", so drop one space of
        # padding to keep a hand-aligned "=" column lined up. A single space
        # is left alone -- that is normal spacing, not alignment.
        pre = match.group("pre")
        if len(pre) > 1:
            pre = pre[:-1]
        return (
            f"{match.group('indent')}scheme{pre}{separator}"
            f"{match.group('post')}\"{scheme}\"{match.group('trail')}"
        )

    new_text, count = pattern.subn(substitute, text)
    return new_text, count


def _rename_key_in_text(text: str, fmt: str, old: str, new: str):
    """Rewrite an assignment's *key*, leaving its value and layout alone.

    Simpler than :func:`_replace_daalg_in_text` because only the name on the
    left of the separator moves; the value can be a multi-line list, an inline
    table or anything else and never has to be understood.

    Returns ``(new_text, count)``.
    """
    separator = "=" if fmt == "toml" else ":"
    pattern = re.compile(
        rf"^(?P<indent>[^\S\n]*){re.escape(old)}(?P<pre>[^\S\n]*){re.escape(separator)}",
        re.MULTILINE,
    )

    def substitute(match):
        # Keep a hand-aligned separator column: absorb the length difference
        # into the padding when there is padding to absorb.
        pre = match.group("pre")
        if len(pre) > 1:
            pre = pre[: max(1, len(pre) - (len(new) - len(old)))]
        return f"{match.group('indent')}{new}{pre}{separator}"

    return pattern.subn(substitute, text)


def _load(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with open(path, "rb") as handle:
            return tomli.load(handle), "toml"
    if suffix in (".yaml", ".yml"):
        with open(path) as handle:
            return yaml.safe_load(handle), "yaml"
    raise ValueError(
        f"Cannot migrate '{path}': only .toml and .yaml/.yml are supported. "
        f"Convert legacy .pipt/.popt files first with `pet convert`."
    )


def _dump(config: dict, path: Path, fmt: str) -> None:
    if fmt == "toml":
        with open(path, "wb") as handle:
            tomli_w.dump(config, handle)
    else:
        with open(path, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)


def migrate_config(path, *, dry_run: bool = False, backup: bool = True) -> MigrationReport:
    """Migrate a config file to the current schema.

    Parameters
    ----------
    path : str or Path
        Path to a ``.toml`` or ``.yaml`` config file.
    dry_run : bool, optional
        Report what would change without writing anything.
    backup : bool, optional
        Keep the original alongside the migrated file as ``<name>.bak``.

    Returns
    -------
    MigrationReport
    """
    path = Path(path)
    config, fmt = _load(path)
    report = MigrationReport()

    if not isinstance(config, dict):
        raise ValueError(f"'{path}' does not contain a mapping at the top level.")

    # Parse first: the parsed value is the reliable source for *what* the new
    # scheme should be, and for the ambiguity warnings.
    schemes = []
    renames = []
    for name in _DA_SECTIONS:
        section = config.get(name)
        if not isinstance(section, dict):
            continue
        had_daalg = "daalg" in section
        present = [old for old in _RENAMED_KEYS if old in section]
        if not had_daalg and not present:
            continue

        before = len(report.changes)
        migrate_section(section, report)
        if len(report.changes) == before:
            continue
        if had_daalg and "scheme" in section:
            schemes.append(section["scheme"])
        renames += [(old, _RENAMED_KEYS[old]) for old in present if _RENAMED_KEYS[old] in section]

    if not report.changed or dry_run:
        return report

    # Write via a surgical text edit so comments, commented-out blocks,
    # indentation, inline tables and quote style all survive.
    original = path.read_text()
    new_text = original
    surgical = True

    if schemes:
        if len(schemes) == 1:
            new_text, count = _replace_daalg_in_text(new_text, fmt, schemes[0])
            surgical = surgical and count == 1
        else:
            # Several sections carry a daalg; one substitution cannot serve both.
            surgical = False

    for old, new in renames:
        new_text, count = _rename_key_in_text(new_text, fmt, old, new)
        surgical = surgical and count > 0

    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    if surgical:
        path.write_text(new_text)
    else:
        # Unusual layout (or several sections): fall back to a full rewrite,
        # but say so -- this is the path that loses comments.
        report.warnings.append(
            "Could not edit every changed line in place, so the file was "
            "rewritten from its parsed contents. Comments, commented-out "
            "blocks and original formatting have been lost; the previous "
            "version is in the .bak file."
        )
        _dump(config, path, fmt)

    return report
