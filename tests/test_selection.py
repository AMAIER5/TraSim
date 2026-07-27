"""
tests/test_selection.py

Tests for population selection.
"""

from __future__ import annotations

from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population import (
    Population,
)
from optimization.selection import (
    Selection,
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


def test_selection_keeps_best_candidates():

    population = Population(
        (
            create_candidate(10.0),
            create_candidate(20.0),
            create_candidate(30.0),
        )
    )

    scores = {
        population[0]: 3.0,
        population[1]: 1.0,
        population[2]: 2.0,
    }

    selection = Selection()

    result = selection.select(
        population,
        scores,
        count=2,
    )

    assert len(result) == 2

    assert result[0] == population[1]

    assert result[1] == population[2]


def test_selection_rejects_too_large_count():

    population = Population(
        (
            create_candidate(10.0),
        )
    )

    selection = Selection()

    try:

        selection.select(
            population,
            {
                population[0]: 1.0,
            },
            count=2,
        )

        assert False

    except ValueError:

        assert True


def test_selection_does_not_modify_population():

    population = Population(
        (
            create_candidate(10.0),
            create_candidate(20.0),
        )
    )

    selection = Selection()

    selection.select(
        population,
        {
            population[0]: 1.0,
            population[1]: 2.0,
        },
        count=1,
    )

    assert len(population) == 2