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

    assert len(parameters) == 4

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