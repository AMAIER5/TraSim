"""
tests/test_csv_parameter_factory.py

Tests for creating optimization parameters
from CSV-based mechanism definitions.
"""

from __future__ import annotations

from mechanism_io import CsvReader

from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)


def test_create_parameter_template_from_csv(
    example_mechanism_csv,
):
    """
    CSV mechanism definition creates
    matching optimization parameters.
    """

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    parameters = CsvParameterFactory.create(
        definition,
    )

    # Each lever creates:
    # - length parameter
    # - angle parameter
    # - pivot.x / pivot.y / pivot.z parameters
    assert len(parameters) == 20

    lever_1 = parameters.get(
        "lever.1.length",
    )

    assert lever_1.minimum == 40
    assert lever_1.maximum == 100
    assert lever_1.value == 60


    lever_2 = parameters.get(
        "lever.2.length",
    )

    assert lever_2.minimum == 30
    assert lever_2.maximum == 90
    assert lever_2.value == 45

    lever_1_angle = parameters.get(
        "lever.1.angle",
    )

    definition_lever_1 = definition.get_lever(1)

    assert lever_1_angle.minimum == definition_lever_1.angle_min
    assert lever_1_angle.maximum == definition_lever_1.angle_max
    assert lever_1_angle.value == definition_lever_1.angle_start



def test_parameter_names_match_lever_ids(
    example_mechanism_csv,
):
    """
    Generated parameter names correspond
    to CSV lever identifiers.
    """

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    parameters = CsvParameterFactory.create(
        definition,
    )

    values = parameters.values()

    assert "lever.1.length" in values
    assert "lever.2.length" in values
    assert "lever.3.length" in values
    assert "lever.4.length" in values

    assert "lever.1.angle" in values
    assert "lever.2.angle" in values
    assert "lever.3.angle" in values
    assert "lever.4.angle" in values

def test_pivot_parameters_free_by_default(
    example_mechanism_csv,
):
    """
    Without pivot_bounds every lever receives
    pivot.x/pivot.y parameters spanning the default
    symmetric search space around the CSV value:
    pivots are optimizable.
    """
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    parameters = CsvParameterFactory.create(
        definition,
    )
    for lever in definition.levers:
        pivot_x = parameters.get(
            f"lever.{lever.id}.pivot.x",
        )
        pivot_y = parameters.get(
            f"lever.{lever.id}.pivot.y",
        )
        pivot_z = parameters.get(
            f"lever.{lever.id}.pivot.z",
        )
        assert pivot_x.minimum < pivot_x.maximum
        assert pivot_x.value == lever.pivot.x
        assert not pivot_x.is_fixed
        assert pivot_y.minimum < pivot_y.maximum
        assert pivot_y.value == lever.pivot.y
        assert not pivot_y.is_fixed
        assert pivot_z.minimum < pivot_z.maximum
        assert pivot_z.value == lever.pivot.z
        assert not pivot_z.is_fixed


def test_pivot_bounds_zero_fixes_pivots(
    example_mechanism_csv,
):
    """
    pivot_bounds=(0, 0) nails pivots down at the
    CSV value (minimum == maximum).
    """
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    parameters = CsvParameterFactory.create(
        definition,
        pivot_bounds=(0.0, 0.0),
    )
    for lever in definition.levers:
        pivot_x = parameters.get(
            f"lever.{lever.id}.pivot.x",
        )
        assert pivot_x.is_fixed
        assert pivot_x.value == lever.pivot.x


def test_mixer_lever_gets_no_pivot_parameters(
):
    """
    A lever with pivot_on (mixer lever) has no pivot
    of its own: no pivot.x/pivot.y parameters are
    created for it.
    """
    import csv

    from pathlib import Path

    from optimization.csv_parameter_factory import (
        DEFAULT_PIVOT_OFFSET,
    )

    mixer_csv = (
        "id,length_min,length_max,"
        "length_start,angle_min,"
        "angle_max,angle_start,"
        "pivot_x,pivot_y,pivot_z,"
        "axis_x,axis_y,axis_z,"
        "driver,coupled,pivot_on\n"
        "1,40,100,60,-40,40,0,"
        "0,60,0,0,0,1,,,lever3\n"
        "2,20,80,50,-60,60,0,"
        "50,10,0,0,0,1,1,,\n"
        "3,40,200,100,-30,30,0,"
        "180,0,0,0,0,1,2,,\n"
    )
    parameters = CsvParameterFactory.create(
        CsvReader.read_mechanism(
            _write_tmp(mixer_csv)
        ),
    )
    names = {
        parameter.name
        for parameter in parameters.parameters
    }
    assert "lever.1.length" in names
    assert "lever.1.angle" in names
    assert not any(
        name.startswith("lever.1.pivot")
        for name in names
    )
    assert "lever.3.pivot.x" in names
    pivot = parameters.get("lever.3.pivot.x")
    assert pivot.minimum == 160.0
    assert pivot.maximum == 200.0


def _write_tmp(text):
    import tempfile

    from pathlib import Path

    directory = tempfile.mkdtemp()
    path = Path(directory) / "mixer.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_pivot_offset_used(
    example_mechanism_csv,
):
    """
    The default search space spans +/- DEFAULT_PIVOT_OFFSET
    around the CSV pivot value.
    """
    from optimization.csv_parameter_factory import (
        DEFAULT_PIVOT_OFFSET,
    )

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    parameters = CsvParameterFactory.create(
        definition,
    )
    for lever in definition.levers:
        pivot_x = parameters.get(
            f"lever.{lever.id}.pivot.x",
        )
        assert pivot_x.minimum == (
            lever.pivot.x - DEFAULT_PIVOT_OFFSET
        )
        assert pivot_x.maximum == (
            lever.pivot.x + DEFAULT_PIVOT_OFFSET
        )


def test_pivot_bounds_free_pivots(
    example_mechanism_csv,
):
    """
    With pivot_bounds the pivot parameters span
    the symmetric box around the CSV pivot, so
    pivots become optimizable.
    """
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    parameters = CsvParameterFactory.create(
        definition,
        pivot_bounds=(-20.0, 20.0),
    )
    for lever in definition.levers:
        pivot_x = parameters.get(
            f"lever.{lever.id}.pivot.x",
        )
        pivot_y = parameters.get(
            f"lever.{lever.id}.pivot.y",
        )
        assert pivot_x.minimum == (
            lever.pivot.x - 20.0
        )
        assert pivot_x.maximum == (
            lever.pivot.x + 20.0
        )
        assert pivot_x.value == lever.pivot.x
        assert not pivot_x.is_fixed
        assert pivot_y.minimum == (
            lever.pivot.y - 20.0
        )
        assert pivot_y.maximum == (
            lever.pivot.y + 20.0
        )
        assert pivot_y.value == lever.pivot.y
        assert not pivot_y.is_fixed


def test_invalid_pivot_bounds_rejected(
    example_mechanism_csv,
):
    """
    pivot_bounds with minimum greater than maximum
    are rejected.
    """
    import pytest

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    with pytest.raises(ValueError):
        CsvParameterFactory.create(
            definition,
            pivot_bounds=(20.0, -20.0),
        )
