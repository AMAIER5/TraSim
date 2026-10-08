from __future__ import annotations

import pytest

from mechanism_io import CsvReader, CsvWriter
from model.lever_definition import (
    LeverDefinition,
    parse_pivot_reference,
)
from model.mechanism_definition import MechanismDefinition

PIVOT_ON_CSV = """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled,pivot_on
1,40,100,60,-40,40,0,0,0,0,0,0,1,,,lever3@180
2,30,90,45,-60,60,0,100,0,0,0,0,1,1,,
3,20,70,35,-30,30,0,200,20,0,0,0,1,2,,
"""


def test_parse_pivot_reference_valid():
    reference = parse_pivot_reference("lever3@180")
    assert reference.lever_id == 3
    assert reference.angle_deg == 180.0


@pytest.mark.parametrize(
    "value",
    [
        "lever3",
        "lever3@",
        "@180",
        "lever3@18a0",
        "lever-three@180",
        "lever3@180@0",
    ],
)
def test_parse_pivot_reference_invalid(value):
    with pytest.raises(ValueError, match="invalid pivot_on"):
        parse_pivot_reference(value)


def test_pivot_reference_property():
    lever = LeverDefinition(
        id=2,
        pivot=None,
        length_min=30,
        length_max=90,
        length_start=45,
        angle_min=-60,
        angle_max=60,
        angle_start=0,
        axis=None,
        pivot_on="lever3@180",
    )
    assert lever.pivot_reference is not None
    assert lever.pivot_reference.lever_id == 3
    assert lever.pivot_reference.angle_deg == 180.0


def test_pivot_reference_property_none():
    lever = LeverDefinition(
        id=1,
        pivot=None,
        length_min=30,
        length_max=90,
        length_start=45,
        angle_min=-60,
        angle_max=60,
        angle_start=0,
        axis=None,
    )
    assert lever.pivot_on is None
    assert lever.pivot_reference is None


def test_read_mechanism_with_pivot_on(tmp_path):
    path = tmp_path / "mechanism.csv"
    path.write_text(PIVOT_ON_CSV, encoding="utf-8")
    mechanism = CsvReader.read_mechanism(path)
    assert mechanism.levers[0].pivot_on == "lever3@180"
    assert mechanism.levers[1].pivot_on is None
    assert mechanism.levers[2].pivot_on is None


def test_read_mechanism_invalid_pivot_on(tmp_path):
    path = tmp_path / "mechanism.csv"
    path.write_text(
        PIVOT_ON_CSV.replace("lever3@180", "lever3@18a0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid pivot_on"):
        CsvReader.read_mechanism(path)


def test_pivot_on_roundtrip(tmp_path):
    path = tmp_path / "mechanism.csv"
    path.write_text(PIVOT_ON_CSV, encoding="utf-8")
    original = CsvReader.read_mechanism(path)
    output = tmp_path / "roundtrip.csv"
    CsvWriter.write_mechanism(original, output)
    restored = CsvReader.read_mechanism(output)
    assert restored == original
    lines = output.read_text(
        encoding="utf-8"
    ).splitlines()
    assert lines[0].endswith("pivot_on")
    assert lines[1].endswith("lever3@180")


def test_write_pivot_on_last_column(tmp_path):
    path = tmp_path / "mechanism.csv"
    path.write_text(PIVOT_ON_CSV, encoding="utf-8")
    mechanism = CsvReader.read_mechanism(path)
    output = tmp_path / "out.csv"
    CsvWriter.write_mechanism(mechanism, output)
    header = (
        output.read_text(encoding="utf-8")
        .splitlines()[0]
        .split(",")
    )
    assert header[-1] == "pivot_on"
    assert header[-2] == "coupled"
