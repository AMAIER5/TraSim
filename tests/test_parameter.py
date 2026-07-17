"""
tests/test_parameter.py

Tests for optimization parameters.
"""

from __future__ import annotations

import pytest

from optimization.parameter import (
    Parameter,
)


def test_parameter_creation():

    parameter = Parameter(
        name="rod_length",
        minimum=10.0,
        maximum=100.0,
        value=50.0,
    )

    assert parameter.name == "rod_length"

    assert parameter.value == 50.0


def test_parameter_rejects_value_below_minimum():

    with pytest.raises(
        ValueError
    ):

        Parameter(
            name="length",
            minimum=10.0,
            maximum=100.0,
            value=5.0,
        )


def test_parameter_rejects_value_above_maximum():

    with pytest.raises(
        ValueError
    ):

        Parameter(
            name="length",
            minimum=10.0,
            maximum=100.0,
            value=120.0,
        )


def test_parameter_minimum_must_be_smaller():

    with pytest.raises(
        ValueError
    ):

        Parameter(
            name="length",
            minimum=100.0,
            maximum=10.0,
            value=50.0,
        )


def test_parameter_is_immutable():

    parameter = Parameter(
        name="length",
        minimum=10.0,
        maximum=100.0,
        value=50.0,
    )

    with pytest.raises(
        AttributeError
    ):

        parameter.value = 60.0