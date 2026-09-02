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
    assert len(parameters) == 8

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