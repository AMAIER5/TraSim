"""
tests/test_parameter.py

Tests for optimization parameters.

Issue #19: Added tests for minimum == maximum (fixed
parameters) and the new is_fixed / range properties.
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


# ---------------------------------------------------------------------------
# Issue #19: Fixed parameters (minimum == maximum)
# ---------------------------------------------------------------------------

def test_parameter_accepts_min_equal_max():
    """
    Issue #19: A parameter with minimum == maximum is
    a fixed (non-optimizable) parameter and must be
    accepted.
    """

    parameter = Parameter(
        name="fixed_length",
        minimum=50.0,
        maximum=50.0,
        value=50.0,
    )

    assert parameter.minimum == 50.0
    assert parameter.maximum == 50.0
    assert parameter.value == 50.0


def test_fixed_parameter_is_fixed_property():
    """
    Issue #19: is_fixed returns True when min == max.
    """

    fixed = Parameter(
        name="fixed",
        minimum=50.0,
        maximum=50.0,
        value=50.0,
    )

    assert fixed.is_fixed is True


def test_non_fixed_parameter_is_not_fixed():
    """
    Issue #19: is_fixed returns False when min < max.
    """

    variable = Parameter(
        name="variable",
        minimum=10.0,
        maximum=100.0,
        value=50.0,
    )

    assert variable.is_fixed is False


def test_fixed_parameter_range_is_zero():
    """
    Issue #19: range property is zero for fixed params.
    """

    fixed = Parameter(
        name="fixed",
        minimum=50.0,
        maximum=50.0,
        value=50.0,
    )

    assert fixed.range == 0.0


def test_variable_parameter_range():
    """
    Issue #19: range property returns max - min.
    """

    variable = Parameter(
        name="variable",
        minimum=10.0,
        maximum=100.0,
        value=50.0,
    )

    assert variable.range == 90.0


def test_fixed_parameter_wrong_value_raises():
    """
    Issue #19: When min == max, the value must equal
    that same number.
    """

    with pytest.raises(ValueError, match="outside parameter range"):
        Parameter(
            name="fixed",
            minimum=50.0,
            maximum=50.0,
            value=51.0,
        )


def test_fixed_parameter_wrong_value_below_raises():
    """
    Issue #19: Value below the fixed point raises.
    """

    with pytest.raises(ValueError, match="outside parameter range"):
        Parameter(
            name="fixed",
            minimum=50.0,
            maximum=50.0,
            value=49.0,
        )


def test_parameter_min_greater_than_max_still_raises():
    """
    Issue #19: minimum > maximum must still raise.
    """

    with pytest.raises(ValueError):
        Parameter(
            name="length",
            minimum=100.0,
            maximum=10.0,
            value=50.0,
        )


def test_fixed_parameter_csv_fixture_scenario():
    """
    Issue #19: Simulates the CSV fixture scenario where
    length_min == length_max == 50 (a fixed lever).
    """

    parameter = Parameter(
        name="lever.1.length",
        minimum=50,
        maximum=50,
        value=50,
    )

    assert parameter.value == 50
    assert parameter.is_fixed
    assert parameter.range == 0.0