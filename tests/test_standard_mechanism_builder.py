"""
tests/test_standard_mechanism_builder.py

Tests for standard mechanism builder.
"""

from __future__ import annotations

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)
from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)


def create_parameters():

    return ParameterSet(
        (
            Parameter(
                name="input_lever_length",
                minimum=10.0,
                maximum=100.0,
                value=40.0,
            ),
            Parameter(
                name="output_lever_length",
                minimum=10.0,
                maximum=100.0,
                value=30.0,
            ),
            Parameter(
                name="rod_length",
                minimum=20.0,
                maximum=200.0,
                value=120.0,
            ),
        )
    )


def test_builder_creates_mechanism():

    builder = StandardMechanismBuilder()

    mechanism = builder.build(
        create_parameters()
    )

    assert len(
        mechanism.stages
    ) == 1


def test_builder_uses_parameters():

    builder = StandardMechanismBuilder()

    mechanism = builder.build(
        create_parameters()
    )

    stage = mechanism.stages[0]

    assert stage.rod_length == 120.0