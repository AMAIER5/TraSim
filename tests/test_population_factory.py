"""
tests/test_population_factory.py

Tests for population factory.
"""

from __future__ import annotations

import random

import pytest

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.population_factory import (
    PopulationFactory,
)


def create_template():

    return ParameterSet(
        (
            Parameter(
                name="length",
                minimum=10.0,
                maximum=100.0,
                value=50.0,
            ),
        )
    )


def test_factory_creates_population():

    factory = PopulationFactory(
        random_generator=random.Random(1)
    )

    population = factory.create(
        create_template(),
        size=10,
    )

    assert len(population) == 10


def test_factory_keeps_parameter_bounds():

    factory = PopulationFactory(
        random_generator=random.Random(1)
    )

    population = factory.create(
        create_template(),
        size=20,
    )

    for candidate in population:

        parameter = candidate.get(
            "length"
        )

        assert (
            10.0
            <= parameter.value
            <= 100.0
        )


def test_factory_does_not_modify_template():

    template = create_template()

    factory = PopulationFactory(
        random_generator=random.Random(1)
    )

    factory.create(
        template,
        size=5,
    )

    assert template.get(
        "length"
    ).value == 50.0


def test_factory_rejects_invalid_size():

    factory = PopulationFactory()

    with pytest.raises(
        ValueError
    ):

        factory.create(
            create_template(),
            size=0,
        )