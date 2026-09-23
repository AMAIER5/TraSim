"""tests/test_csv_convention.py

Tests for the international and German CSV
conventions (column and decimal separators).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from analysis.target_curve import (
    DiscreteTargetCurve,
    TargetCurve,
)
from mechanism_io.csv_convention import (
    GERMAN,
    INTERNATIONAL,
    detect_convention,
)
from mechanism_io.csv_reader import CsvReader
from mechanism_io.csv_writer import CsvWriter

GERMAN_MECHANISM_CSV = (
    "id;length_min;length_max;length_start;"
    "angle_min;angle_max;angle_start;"
    "pivot_x;pivot_y;pivot_z;"
    "axis_x;axis_y;axis_z;driver;coupled\n"
    "1;40;100;60,5;-40;40;0;0;0;0;0;0;1;;\n"
    "2;30,5;90;45;-60;60;0;100,25;0;0;0;0;1;1;\n"
)

GERMAN_TARGET_CSV = """\
input_angle;output_angle
-40;-30
0;0
40;30
"""


@pytest.fixture
def german_mechanism_csv(tmp_path):
    path = tmp_path / "mechanism_de.csv"
    path.write_text(
        GERMAN_MECHANISM_CSV,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def german_target_csv(tmp_path):
    path = tmp_path / "target_de.csv"
    path.write_text(
        GERMAN_TARGET_CSV,
        encoding="utf-8",
    )
    return path


def test_detect_convention_international(
    tmp_path,
):
    path = tmp_path / "file.csv"
    path.write_text(
        "a,b,c\n1,2,3\n",
        encoding="utf-8",
    )
    assert detect_convention(path) == INTERNATIONAL


def test_detect_convention_german(tmp_path):
    path = tmp_path / "file.csv"
    path.write_text(
        "a;b;c\n1;2;3\n",
        encoding="utf-8",
    )
    assert detect_convention(path) == GERMAN


def test_german_convention_properties():
    assert GERMAN.delimiter == ";"
    assert GERMAN.decimal == ","
    assert GERMAN.is_german
    assert not INTERNATIONAL.is_german


def test_parse_float_german():
    assert GERMAN.parse_float("1,5") == 1.5
    assert GERMAN.parse_float("-12,25") == -12.25
    assert INTERNATIONAL.parse_float(
        "1.5",
    ) == 1.5


def test_format_float_german():
    assert GERMAN.format_float(1.5) == "1,5"
    assert GERMAN.format_float(2) == "2"
    assert INTERNATIONAL.format_float(
        1.5,
    ) == "1.5"


def test_read_german_mechanism(
    german_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        german_mechanism_csv,
    )
    assert CsvReader.last_convention == GERMAN
    lever = definition.levers[0]
    assert lever.length_start == 60.5
    assert lever.pivot.x == 0.0
    second = definition.levers[1]
    assert second.length_min == 30.5
    assert second.pivot.x == 100.25
    assert second.driver == 1


def test_read_international_mechanism(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    assert (
        CsvReader.last_convention
        == INTERNATIONAL
    )
    lever = definition.levers[0]
    assert lever.length_start == 60
    assert lever.length_min == 40


def test_write_and_reread_german_mechanism(
    german_mechanism_csv,
    tmp_path,
):
    definition = CsvReader.read_mechanism(
        german_mechanism_csv,
    )
    output = tmp_path / "out_de.csv"
    CsvWriter.write_mechanism(
        definition,
        output,
        convention=GERMAN,
    )
    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[0].count(
        ";",
    ) > 5
    assert "," in text
    restored = CsvReader.read_mechanism(output)
    assert restored == definition


def test_write_international_mechanism(
    example_mechanism_csv,
    tmp_path,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    output = tmp_path / "out.csv"
    CsvWriter.write_mechanism(
        definition,
        output,
    )
    text = output.read_text(encoding="utf-8")
    first_line = text.splitlines()[0]
    assert ";" not in first_line
    restored = CsvReader.read_mechanism(output)
    assert restored == definition


def test_read_german_target_curve(
    german_target_csv,
):
    curve = TargetCurve.from_csv(
        german_target_csv,
    )
    assert math.degrees(
        curve.evaluate(math.radians(-40)),
    ) == pytest.approx(-30)
    assert math.degrees(
        curve.evaluate(math.radians(0)),
    ) == pytest.approx(0)
    assert math.degrees(
        curve.evaluate(math.radians(40)),
    ) == pytest.approx(30)


def test_read_german_target_curve_strict(
    german_target_csv,
):
    curve = TargetCurve.from_csv_strict(
        german_target_csv,
    )
    assert isinstance(
        curve,
        DiscreteTargetCurve,
    )
    assert len(curve.input_angles) == 3


def test_read_international_target_curve(
    simple_target_csv,
):
    curve = TargetCurve.from_csv(
        simple_target_csv,
    )
    assert math.degrees(
        curve.evaluate(math.radians(0)),
    ) == pytest.approx(0)
