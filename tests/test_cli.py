"""Tests for the `pet` command-line interface."""



from pet_cli.__main__ import main

MINIMAL_PIPT = """\
DATAASSIM

DAALG
esmda\tesmda

DATA
truedata.csv

DATAVAR
var.csv

OBSNAME
obs

ENERGY
0.99

FWDSIM

PARALLEL
1

DATATYPE
pressure

"""


def test_version(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert out.strip()


def test_validate_missing_file(capsys):
    assert main(["validate", "does_not_exist.toml"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_validate_valid_toml(tmp_path, capsys):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[dataassim]\nscheme = "esmda"\ndata = "d.csv"\ndatavar = "v.csv"\n'
        'obsname = "obs"\nenergy = 0.99\n\n[fwdsim]\nparallel = 1\ndatatype = ["pressure"]\n'
    )
    assert main(["validate", str(config_file)]) == 0
    assert "No problems found." in capsys.readouterr().out


def test_validate_reports_missing_mandatory_keyword(tmp_path, capsys):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[fwdsim]\nparallel = 1\n')
    assert main(["validate", str(config_file)]) == 1
    assert "[simulator] datatype: required" in capsys.readouterr().out


def test_convert_pipt_to_toml(tmp_path, capsys):
    pipt_file = tmp_path / "case.pipt"
    pipt_file.write_text(MINIMAL_PIPT)

    assert main(["convert", str(pipt_file), "--to", "toml"]) == 0

    toml_file = tmp_path / "case.toml"
    assert toml_file.is_file()
    assert "Wrote" in capsys.readouterr().out


def test_convert_pipt_to_yaml(tmp_path):
    pipt_file = tmp_path / "case.pipt"
    pipt_file.write_text(MINIMAL_PIPT)

    assert main(["convert", str(pipt_file), "--to", "yaml"]) == 0
    assert (tmp_path / "case.yaml").is_file()
