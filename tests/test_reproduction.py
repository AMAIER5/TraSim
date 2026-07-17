"""
tests/test_reproduction.py

Tests for reproduction of candidates.
"""

from __future__ import annotations

import random

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_mutation import (
    ParameterMutation,
)

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.population import (
    Population,
)

from optimization.reproduction import (
    Reproduction,
)


def create_candidate(
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


def test_reproduction_creates_children():

    population = Population(
        (
            create_candidate(50.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(1)
        )
    )

    children = reproduction.create(
        population,
        count=3,
    )

    assert len(children) == 3


def test_reproduction_keeps_parameters_valid():

    population = Population(
        (
            create_candidate(50.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(2)
        )
    )

    children = reproduction.create(
        population,
        count=10,
    )

    for child in children:

        parameter = child.get(
            "length"
        )

        assert (
            parameter.minimum
            <=
            parameter.value
            <=
            parameter.maximum
        )


def test_reproduction_does_not_change_parent():

    parent = create_candidate(
        50.0
    )

    population = Population(
        (
            parent,
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(1)
        )
    )

    reproduction.create(
        population,
        count=1,
    )

    assert parent.get(
        "length"
    ).value == 50.0