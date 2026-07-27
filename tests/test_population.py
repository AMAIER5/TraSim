"""
tests/test_population.py

Tests for optimization populations.
"""

from __future__ import annotations

import pytest

from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population import (
    Population,
)


def create_parameter_set(
    value: float,
):

    return ParameterSet(
        (
            Parameter(
                name="length",
                minimum=10.0,
                maximum=100.0,
                value=value,
            ),
        )
    )


def test_population_creation():

    population = Population(
        (
            create_parameter_set(20.0),
            create_parameter_set(40.0),
        )
    )

    assert len(
        population
    ) == 2


def test_population_access():

    population = Population(
        (
            create_parameter_set(20.0),
        )
    )

    assert population[0].get(
        "length"
    ).value == 20.0


def test_population_is_iterable():

    population = Population(
        (
            create_parameter_set(20.0),
            create_parameter_set(40.0),
        )
    )

    values = [
        item.get("length").value
        for item in population
    ]

    assert values == [
        20.0,
        40.0,
    ]


def test_empty_population_is_rejected():

    with pytest.raises(
        ValueError
    ):

        Population(
            ()
        )


def test_population_is_immutable():

    population = Population(
        (
            create_parameter_set(20.0),
        )
    )

    with pytest.raises(
        AttributeError
    ):

        population.members = ()