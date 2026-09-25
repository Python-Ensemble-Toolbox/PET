"""Tests for `pet migrate` and the config-schema change it implements."""

import pytest

from pet_cli.__main__ import main
from pet_cli.migrate import MigrationReport, migrate_config, migrate_section

LEGACY_TOML = """\
[dataassim]
daalg = ["esmda", "esmda"]
analysis = "approx"
energy = 0.99

[fwdsim]
parallel = 1
datatype = ["pressure"]
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


# ----------------------------------------------------------------------
# Section-level migration
# ----------------------------------------------------------------------

def test_section_daalg_to_scheme():
    report = MigrationReport()
    section = migrate_section({"daalg": ["esmda", "esmda"], "analysis": "approx"}, report)
    assert section["scheme"] == "esmda"
    assert "daalg" not in section
    assert report.changed


def test_section_keeps_second_entry_and_warns_on_mismatch():
    """The second entry is the one that selected the class historically."""
    report = MigrationReport()
    section = migrate_section({"daalg": ["enrml", "lmenrml"], "analysis": "full"}, report)
    assert section["scheme"] == "lmenrml"
    assert any("differing entries" in w for w in report.warnings)


def test_section_accepts_bare_string():
    report = MigrationReport()
    assert migrate_section({"daalg": "esmda", "analysis": "approx"}, report)["scheme"] == "esmda"


def test_section_warns_when_analysis_missing():
    report = MigrationReport()
    migrate_section({"daalg": ["esmda", "esmda"]}, report)
    assert any("analysis" in w for w in report.warnings)


def test_section_without_daalg_is_untouched():
    report = MigrationReport()
    section = migrate_section({"scheme": "esmda", "analysis": "approx"}, report)
    assert section == {"scheme": "esmda", "analysis": "approx"}
    assert not report.changed


def test_section_leaves_unexpected_daalg_alone():
    report = MigrationReport()
    section = migrate_section({"daalg": 42}, report)
    assert section["daalg"] == 42
    assert report.warnings


# ----------------------------------------------------------------------
# File-level migration
# ----------------------------------------------------------------------

def test_migrate_toml_writes_backup(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    report = migrate_config(path)
    assert report.changed
    assert (tmp_path / "case.toml.bak").exists()
    assert "scheme" in path.read_text()
    assert "daalg" not in path.read_text()


def test_migrate_preserves_other_keys(tmp_path):
    import tomli

    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    migrate_config(path)
    with open(path, "rb") as handle:
        cfg = tomli.load(handle)
    assert cfg["dataassim"]["analysis"] == "approx"
    assert cfg["dataassim"]["energy"] == 0.99
    assert cfg["fwdsim"]["datatype"] == ["pressure"]


def test_migrate_dry_run_writes_nothing(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    before = path.read_text()
    report = migrate_config(path, dry_run=True)
    assert report.changed
    assert path.read_text() == before
    assert not (tmp_path / "case.toml.bak").exists()


def test_migrate_is_idempotent(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    migrate_config(path)
    assert not migrate_config(path).changed


def test_migrate_yaml(tmp_path):
    import yaml

    path = _write(
        tmp_path, "case.yaml",
        "dataassim:\n  daalg: [esmda, esmda]\n  analysis: approx\n",
    )
    migrate_config(path)
    cfg = yaml.safe_load(path.read_text())
    assert cfg["dataassim"]["scheme"] == "esmda"


def test_migrate_rejects_unsupported_format(tmp_path):
    path = _write(tmp_path, "case.pipt", "DATAASSIM\n")
    with pytest.raises(ValueError, match="only .toml and .yaml"):
        migrate_config(path)


def test_migrate_handles_optim_section(tmp_path):
    path = _write(tmp_path, "case.toml", '[optim]\ndaalg = ["esmda", "esmda"]\nanalysis = "approx"\n')
    assert migrate_config(path).changed


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_cli_migrate(tmp_path, capsys):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    assert main(["migrate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "scheme = 'esmda'" in out
    assert ".bak" in out


def test_cli_migrate_no_backup(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    assert main(["migrate", str(path), "--no-backup"]) == 0
    assert not (tmp_path / "case.toml.bak").exists()


def test_cli_migrate_missing_file(capsys):
    assert main(["migrate", "nope.toml"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_cli_migrate_already_current(tmp_path, capsys):
    path = _write(tmp_path, "case.toml", '[dataassim]\nscheme = "esmda"\nanalysis = "approx"\n')
    assert main(["migrate", str(path)]) == 0
    assert "already on the current schema" in capsys.readouterr().out


# ----------------------------------------------------------------------
# init_da must point users at the tool rather than failing cryptically
# ----------------------------------------------------------------------

def test_init_da_rejects_legacy_daalg_with_migration_hint():
    from pipt import pipt_init

    with pytest.raises(ValueError, match="pet migrate"):
        pipt_init.init_da({"daalg": ["esmda", "esmda"], "analysis": "approx"}, {}, None)


def test_init_da_accepts_new_scheme_key():
    from pipt import pipt_init
    from pipt.update_schemes import registry

    class Spy:
        def __init__(self, da, en, sim):
            self.ok = True

    registry.register_scheme("spy", "approx", Spy)
    try:
        obj = pipt_init.init_da({"scheme": "spy", "analysis": "approx"}, {}, None)
        assert obj.ok
    finally:
        registry.SPECIAL_SCHEMES.pop(("spy", "approx"), None)


def test_init_da_rejects_non_string_scheme():
    from pipt import pipt_init

    with pytest.raises(ValueError, match="as a string"):
        pipt_init.init_da({"scheme": ["esmda"], "analysis": "approx"}, {}, None)


# ----------------------------------------------------------------------
# Formatting preservation
#
# A round trip through a TOML/YAML writer discards everything that is not
# data. Real configs carry comments, commented-out alternative blocks,
# hand-aligned columns and inline tables, so the migration must edit the
# daalg line in place instead.
# ----------------------------------------------------------------------

REALISTIC_TOML = """\
[ensemble]
    ne = 100
    state = "PORO"
    prior_PORO  = {var=1.0, grid=[50, 50]} # var is used for scaling

[dataassim]
    daalg       = ["enrml", "gnenrml"]
    energy      = 99
    analysis    = "approx"

    # Distance-based localization options
    #[dataassim.localization]
    #    name    = "distance_loc"
    #    field   = [1, 50, 50]   # nz, nx, ny

    [dataassim.localization]
        name   = "autoadaloc"
        field  = [50, 50]   # nx, ny
"""


def test_migration_changes_exactly_one_line(tmp_path):
    path = _write(tmp_path, "case.toml", REALISTIC_TOML)
    migrate_config(path)

    before = REALISTIC_TOML.split("\n")
    after = path.read_text().split("\n")
    assert len(before) == len(after), "line count changed; the file was rewritten"

    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1, f"expected 1 changed line, got {len(differing)}"
    assert "daalg" in before[differing[0]]
    assert 'scheme' in after[differing[0]]


def test_comments_and_commented_out_blocks_survive(tmp_path):
    path = _write(tmp_path, "case.toml", REALISTIC_TOML)
    migrate_config(path)
    text = path.read_text()

    assert "# var is used for scaling" in text
    assert "# Distance-based localization options" in text
    assert '#    name    = "distance_loc"' in text, "commented-out block was deleted"
    assert "# nx, ny" in text


def test_inline_table_and_indentation_survive(tmp_path):
    path = _write(tmp_path, "case.toml", REALISTIC_TOML)
    migrate_config(path)
    text = path.read_text()

    assert "prior_PORO  = {var=1.0, grid=[50, 50]}" in text, "inline table was expanded"
    assert "    energy      = 99" in text, "indentation was flattened"


def test_aligned_equals_column_is_kept(tmp_path):
    """`scheme` is one char longer than `daalg`; padding absorbs the difference."""
    path = _write(tmp_path, "case.toml", REALISTIC_TOML)
    migrate_config(path)

    lines = [ln for ln in path.read_text().split("\n") if "=" in ln and "#" not in ln]
    scheme_line = next(ln for ln in lines if "scheme" in ln)
    energy_line = next(ln for ln in lines if "energy" in ln)
    assert scheme_line.index("=") == energy_line.index("=")


def test_only_the_scheme_key_changes_semantically(tmp_path):
    import tomli

    path = _write(tmp_path, "case.toml", REALISTIC_TOML)
    original = tomli.loads(REALISTIC_TOML)
    migrate_config(path)
    with open(path, "rb") as handle:
        migrated = tomli.load(handle)

    assert migrated["dataassim"]["scheme"] == "gnenrml"
    original["dataassim"].pop("daalg")
    migrated["dataassim"].pop("scheme")
    assert original == migrated


def test_single_space_spacing_is_left_alone(tmp_path):
    path = _write(tmp_path, "case.toml", '[dataassim]\ndaalg = ["esmda", "esmda"]\n')
    migrate_config(path)
    assert 'scheme = "esmda"' in path.read_text()


def test_yaml_inline_form_preserves_comments(tmp_path):
    path = _write(
        tmp_path, "case.yaml",
        "dataassim:\n  # which algorithm\n  daalg: [esmda, esmda]\n  analysis: approx\n",
    )
    migrate_config(path)
    text = path.read_text()
    assert "# which algorithm" in text
    assert "scheme:" in text and "daalg" not in text


# ----------------------------------------------------------------------
# analysisdebug -> savedata
# ----------------------------------------------------------------------
def test_analysisdebug_is_renamed_to_savedata(tmp_path):
    path = _write(
        tmp_path, "case.toml",
        '[dataassim]\nscheme = "esmda"\nanalysisdebug = ["state", "pred_data"]\n',
    )
    report = migrate_config(path)

    text = path.read_text()
    assert 'savedata = ["state", "pred_data"]' in text
    assert "analysisdebug" not in text
    assert any("savedata" in change for change in report.changes)


def test_renaming_the_key_preserves_a_multiline_value_and_comments(tmp_path):
    """Only the name left of the separator moves, so the value is never parsed."""
    path = _write(
        tmp_path, "case.toml",
        '[dataassim]\n'
        'scheme = "esmda"\n'
        '# what to record each iteration\n'
        'analysisdebug = [\n'
        '    "state",     # the ensemble\n'
        '    "pred_data",\n'
        ']\n',
    )
    migrate_config(path)
    text = path.read_text()

    assert "# what to record each iteration" in text
    assert "# the ensemble" in text
    assert text.count('"pred_data",\n') == 1
    assert "savedata = [\n" in text


def test_rename_and_daalg_migrate_together(tmp_path):
    path = _write(
        tmp_path, "case.toml",
        '[dataassim]\ndaalg = ["esmda", "esmda"]\nanalysisdebug = ["state"]\n',
    )
    migrate_config(path)
    text = path.read_text()

    assert 'scheme = "esmda"' in text
    assert 'savedata = ["state"]' in text
    assert "daalg" not in text and "analysisdebug" not in text


def test_both_spellings_present_is_reported_not_guessed(tmp_path):
    path = _write(
        tmp_path, "case.toml",
        '[dataassim]\nscheme = "esmda"\nsavedata = ["state"]\nanalysisdebug = ["pred_data"]\n',
    )
    report = migrate_config(path)

    assert any("savedata" in warning for warning in report.warnings)
    assert "analysisdebug" in path.read_text()


def test_yaml_rename_preserves_comments(tmp_path):
    path = _write(
        tmp_path, "case.yaml",
        "dataassim:\n  scheme: esmda\n  # variables to keep\n  analysisdebug: [state]\n",
    )
    migrate_config(path)
    text = path.read_text()

    assert "# variables to keep" in text
    assert "savedata: [state]" in text and "analysisdebug" not in text
