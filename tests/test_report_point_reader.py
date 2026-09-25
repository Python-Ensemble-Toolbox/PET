import pytest
import datetime as dt
import yaml

from input_output.organize import report_point_file_reader

DATETIMES_STR = [
    "2024-01-01 12:00:00",
    "2024-01-02 13:30:00",
    "2024-01-03 14:45:00"
]
DATETIMES_STR_ISO = [
    "2024-01-01T12:00:00",
    "2024-01-02T13:30:00",
    "2024-01-03T14:45:00"
]
DATETIMES = [
    dt.datetime(2024, 1, 1, 12, 0, 0),
    dt.datetime(2024, 1, 2, 13, 30, 0),
    dt.datetime(2024, 1, 3, 14, 45, 0)
]
INDEX = [1, 2, 3]


def test_report_point_file_reader_csv_int(tmp_path):
    # Create a CSV file with integer report points
    csv_content = "\n".join(str(i) for i in INDEX)
    csv_file = tmp_path / "test_int.csv"
    csv_file.write_text(csv_content)
    points = report_point_file_reader(str(csv_file))
    assert all(isinstance(val, int) for val in points)
    assert points == INDEX


def test_report_point_file_reader_csv_iso(tmp_path):
    # Create a CSV file datetimes (ISO)
    csv_content = "\n".join(DATETIMES_STR_ISO)
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    points = report_point_file_reader(str(csv_file))
    assert all(isinstance(dt_val, dt.datetime) for dt_val in points)
    assert points == DATETIMES

def test_report_point_file_reader_csv(tmp_path):
    # Create a CSV file datetimes (non-ISO)
    csv_content = "\n".join(DATETIMES_STR)
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    points = report_point_file_reader(str(csv_file))
    assert all(isinstance(dt_val, dt.datetime) for dt_val in points)
    assert points == DATETIMES

def test_report_point_file_reader_txt_iso(tmp_path):
    # Create a TXT file with ISO datetimes
    txt_content = "\n".join(DATETIMES_STR_ISO)
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(txt_content)
    result = report_point_file_reader(str(txt_file))
    assert all(isinstance(dt_val, dt.datetime) for dt_val in result)
    assert result == DATETIMES

def test_report_point_file_reader_txt(tmp_path):
    # Create a TXT file with non-ISO datetimes
    txt_content = "\n".join(DATETIMES_STR)
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(txt_content)
    result = report_point_file_reader(str(txt_file))
    assert all(isinstance(dt_val, dt.datetime) for dt_val in result)
    assert result == DATETIMES

def test_report_point_file_reader_txt_int(tmp_path):
    # Create a TXT file with integer report points
    txt_content = "\n".join(str(i) for i in INDEX)
    txt_file = tmp_path / "test_int.txt"
    txt_file.write_text(txt_content)
    result = report_point_file_reader(str(txt_file))
    assert all(isinstance(val, int) for val in result)
    assert result == INDEX

def test_report_point_file_reader_yaml_iso(tmp_path):
    # Create a YAML file with ISO datetimes
    yaml_content = yaml.dump(DATETIMES_STR_ISO)
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    result = report_point_file_reader(str(yaml_file))
    assert all(isinstance(dt_val, dt.datetime) for dt_val in result)
    assert result == DATETIMES

def test_report_point_file_reader_yaml(tmp_path):
    # Create a YAML file with non-ISO datetimes
    yaml_content = yaml.dump(DATETIMES_STR)
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    result = report_point_file_reader(str(yaml_file))
    assert all(isinstance(dt_val, dt.datetime) for dt_val in result)
    assert result == DATETIMES

def test_report_point_file_reader_yaml_int(tmp_path):
    # Create a YAML file with integer report points
    yaml_content = yaml.dump(INDEX)
    yaml_file = tmp_path / "test_int.yaml"
    yaml_file.write_text(yaml_content)
    result = report_point_file_reader(str(yaml_file))
    assert all(isinstance(val, int) for val in result)
    assert result == INDEX


def test_report_point_file_reader_unsupported(tmp_path):
    # Create an unsupported file type
    other_file = tmp_path / "test.unsupported"
    other_file.write_text("dummy")
    with pytest.raises(ValueError):
        report_point_file_reader(str(other_file))

def test_report_point_file_reader_missing_file():
    with pytest.raises(FileNotFoundError):
        report_point_file_reader("nonexistent.csv")
