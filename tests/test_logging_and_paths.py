"""Library code keeps to its own logger and its own folder."""

import logging
from types import SimpleNamespace

import numpy as np

from ensemble.ensemble import BaseEnsemble
from ensemble.logger import PetLogger
from pipt.ensembles.forecast import ForecastMixin


def test_two_loggers_write_to_their_own_files(tmp_path):
    """A second PetLogger used to log into the first one's file: basicConfig
    configures the root logger once per process and is a no-op afterwards."""
    first = PetLogger(str(tmp_path / "first.log"))
    second = PetLogger(str(tmp_path / "second.log"))
    first("one")
    second("two")
    for handler in first._logger.handlers + second._logger.handlers:
        handler.flush()

    assert "one" in (tmp_path / "first.log").read_text() and "two" not in (tmp_path / "first.log").read_text()
    assert "two" in (tmp_path / "second.log").read_text() and "one" not in (tmp_path / "second.log").read_text()


def test_a_logger_does_not_configure_the_root_logger(tmp_path):
    before = list(logging.getLogger().handlers)
    PetLogger(str(tmp_path / "x.log"))
    assert list(logging.getLogger().handlers) == before


def test_the_log_file_is_written_as_utf8(tmp_path):
    """The timestamp format embeds U+2502 and the tables draw with box characters.
    Opened in the OS default encoding, every record raised UnicodeEncodeError on a
    cp1252 Windows and the file stayed empty while the run carried on regardless."""
    logger = PetLogger(str(tmp_path / "encoding.log"))
    logger("one")
    file_handlers = [h for h in logger._logger.handlers if isinstance(h, logging.FileHandler)]
    for handler in file_handlers:
        handler.flush()

    assert [h.encoding for h in file_handlers] == ["utf-8"]
    assert "│" in (tmp_path / "encoding.log").read_text(encoding="utf-8")


def test_save_folder_is_not_created_by_reading_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    host = object.__new__(ForecastMixin)
    host.keys_da = {"savefolder": "Out"}

    assert host.save_folder == "Out"
    assert not (tmp_path / "Out").exists()          # reading creates nothing ...
    assert host._save_path("a.npz") == "Out/a.npz"
    assert (tmp_path / "Out").is_dir()               # ... writing does


def test_all_members_failing_raises_instead_of_exiting():
    host = SimpleNamespace(logger=SimpleNamespace(info=lambda m: None), save=lambda: None, rng=np.random)
    enX = np.zeros((2, 3))
    try:
        BaseEnsemble._replace_failed_simulations(host, [False, False, False], enX)
    except RuntimeError as err:
        assert "All started simulations failed" in str(err)
    else:
        raise AssertionError("expected a RuntimeError")
