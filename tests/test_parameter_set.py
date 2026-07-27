"""
tests/test_parameter_set.py

Tests for parameter collections.
"""

from __future__ import annotations

import pytest

from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)


def create_parameters():

    return (
        Parameter(
            name="rod_length",
            minimum=10.0,
            maximum=100.0,
            value=50.0,
        ),
        Parameter(
            name="lever_length",
            minimum=20.0,
            maximum=80.0,
            value=40.0,
        ),
    )


def test_parameter_set_creation():

    parameter_set = ParameterSet(
        create_parameters()
    )

    assert len(
        parameter_set
    ) == 2


def test_parameter_access_by_name():

    parameter_set = ParameterSet(
        create_parameters()
    )

    parameter = parameter_set.get(
        "rod_length"
    )

    assert parameter.value == 50.0


def test_unknown_parameter_raises():

    parameter_set = ParameterSet(
        create_parameters()
    )

    with pytest.raises(
        KeyError
    ):

        parameter_set.get(
            "unknown"
        )


def test_duplicate_names_are_rejected():

    parameters = (
        Parameter(
            name="length",
            minimum=1.0,
            maximum=10.0,
            value=5.0,
        ),
        Parameter(
            name="length",
            minimum=1.0,
            maximum=10.0,
            value=6.0,
        ),
    )

    with pytest.raises(
        ValueError
    ):

        ParameterSet(
            parameters
        )


def test_parameter_set_is_immutable():

    parameter_set = ParameterSet(
        create_parameters()
    )

    with pytest.raises(
        AttributeError
    ):

        parameter_set.parameters = ()