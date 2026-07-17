"""
tests/test_mechanism_factory.py

Tests for mechanism factory.
"""

from __future__ import annotations

import pytest

from mechanics.mechanism_factory import (
    MechanismFactory,
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
                name="rod_length",
                minimum=10.0,
                maximum=100.0,
                value=50.0,
            ),
        )
    )


def test_factory_creates_mechanism():

    factory = MechanismFactory(
        builder=lambda parameters:
            "mechanism",
    )

    result = factory.create(
        create_parameters()
    )

    assert result == "mechanism"


def test_factory_passes_parameters():

    received = None

    def builder(parameters):

        nonlocal received

        received = parameters

        return "mechanism"

    factory = MechanismFactory(
        builder=builder,
    )

    factory.create(
        create_parameters()
    )

    assert received is not None

    assert received.get(
        "rod_length"
    ).value == 50.0


def test_factory_requires_builder():

    with pytest.raises(
        TypeError
    ):

        MechanismFactory(
            builder=None
        )