"""One boundary turns whatever a config looks like into the one form PET reads, and says what would fail."""

import pytest

from input_output import config, read_config
from input_output.config import ConfigError


def test_aliases_flags_and_row_blocks_become_canonical_and_the_caller_is_untouched():
    raw = {"scheme": "esmda", "truedata": "d.pkl", "var": "v.pkl", "save_folder": "out", "restartfile": "r.pkl",
           "emp_cov": "yes", "scale_data": "no", "restart": "true", "iteration": [["max_iter", 3], ["lambda", 1.0]],
           "prior_x": [["mean", 1.0], ["var", 2.0]]}
    before = {key: (list(value) if isinstance(value, list) else value) for key, value in raw.items()}

    normalised = config.normalize_dataassim(raw)

    assert normalised["data"] == "d.pkl" and normalised["datavar"] == "v.pkl"
    assert normalised["savefolder"] == "out" and normalised["restart_file"] == "r.pkl"
    assert normalised["emp_cov"] is True and normalised["scale_data"] is False and normalised["restart"] is True
    assert normalised["iteration"] == {"max_iter": 3, "lambda": 1.0}
    assert normalised["prior_x"] == {"mean": 1.0, "var": 2.0}
    assert raw == before                                    # a copy was normalised, not the caller's dict
    assert config.normalize_dataassim(normalised) == normalised   # idempotent


def test_the_canonical_spelling_wins_when_both_are_given():
    assert config.normalize_dataassim({"data": "new.pkl", "truedata": "old.pkl"})["data"] == "new.pkl"
    assert config.normalize_ensemble({"importstaticvar": "a.npz", "save_folder": "f"}) == {"importstate": "a.npz", "savefolder": "f"}


def test_validate_names_the_section_and_key():
    problems = config.validate({"scheme": "esmda"}, {"parallel": 1}, {"state": ["x"]})
    messages = [str(p) for p in problems]
    assert "[dataassim] data: required: the observed data" in messages
    assert "[dataassim] datavar: required: the observation variance" in messages
    assert any(m.startswith("[dataassim] obsname") for m in messages)
    assert any(m.startswith("[simulator] datatype") for m in messages)
    assert any(m.startswith("[ensemble] ne") for m in messages)
    assert any(m.startswith("[ensemble] prior_x") for m in messages)
    fatal = {p.key for p in config.fatal_problems({"scheme": "esmda"}, {"parallel": 1}, {"state": ["x"]})}
    assert fatal == {"data", "datavar", "obsname", "prior_x"}     # `ne` has a default, `datatype` can come from the data file


def test_a_complete_config_has_no_problems_and_unknown_keys_are_pointed_out():
    prb = {"scheme": "esmda", "data": "d.pkl", "datavar": "v.pkl", "obsname": "t", "restartsve": True}
    ens = {"ne": 5, "state": ["x"], "prior_x": {"var": 1.0}}
    assert config.validate(prb, {"datatype": ["x"]}, ens) == []
    assert config.unknown_keys(prb, ens) == ["[dataassim] restartsve"]


def test_every_reader_returns_three_normalised_sections(tmp_path):
    (tmp_path / "c.toml").write_text('[dataassim]\nscheme = "esmda"\ntruedata = "d.pkl"\ndatavar = "v.pkl"\nobsname = "t"\n'
                                     'emp_cov = "yes"\n[fwdsim]\ndatatype = ["x"]\n')
    (tmp_path / "c.yaml").write_text('dataassim:\n  scheme: esmda\n  truedata: d.pkl\n  datavar: v.pkl\n  obsname: t\n'
                                     '  emp_cov: "yes"\nfwdsim:\n  datatype: [x]\n')
    (tmp_path / "c.pipt").write_text("DATAASSIM\n\nSCHEME\nesmda\n\nTRUEDATA\nd.pkl\n\nDATAVAR\nv.pkl\n\nOBSNAME\nt\n\n"
                                     "EMP_COV\nyes\n\nFWDSIM\n\nDATATYPE\nx\n\nPARALLEL\n1\n")
    for name in ("c.toml", "c.yaml", "c.pipt"):
        sections = read_config.read(str(tmp_path / name))
        assert len(sections) == 3, name
        prb, sim, ens = sections
        assert prb["data"] == "d.pkl" and "truedata" not in prb, name
        assert prb["emp_cov"] is True, name
        assert sim["datatype"] == ["x"] and ens == {}, name


def test_building_an_ensemble_reports_what_is_missing_and_leaves_the_config_alone():
    from pipt.ensembles import AssimilationEnsemble
    from simulator.vanderpol import VanDerPolOscillator

    keys_da = {"scheme": "esmda", "obsname": "steps"}     # no data, no variance
    keys_en = {"ne": 4, "state": ["x1"], "prior_x1": {"var": 1.0}}
    with pytest.raises(ConfigError, match=r"\[dataassim\] data: required.*\n.*\[dataassim\] datavar: required"):
        AssimilationEnsemble(keys_da, keys_en, VanDerPolOscillator({"reporttype": "steps", "reportpoints": [1], "datatype": ["x1"]}))
    assert keys_da == {"scheme": "esmda", "obsname": "steps"}
