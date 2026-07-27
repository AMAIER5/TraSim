"""
tests/test_generation.py

Tests for evolutionary generation handling.
"""

from __future__ import annotations

from optimization.generation import (
    Generation,
)
from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population import (
    Population,
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


def test_generation_creates_next_population():

    population = Population(
        (
            create_candidate(20.0),
            create_candidate(40.0),
            create_candidate(60.0),
        )
    )

    generation = Generation(
        population=population,
        evaluator=lambda candidate:
            candidate.get("length").value,
        selection_count=2,
    )

    result = generation.next()

    assert len(result) == 2


def test_generation_keeps_best_candidates():

    population = Population(
        (
            create_candidate(60.0),
            create_candidate(20.0),
            create_candidate(40.0),
        )
    )

    generation = Generation(
        population=population,
        evaluator=lambda candidate:
            candidate.get("length").value,
        selection_count=1,
    )

    result = generation.next()

    assert result[0].get(
        "length"
    ).value == 20.0


def test_generation_does_not_modify_original():

    population = Population(
        (
            create_candidate(20.0),
            create_candidate(40.0),
        )
    )

    generation = Generation(
        population=population,
        evaluator=lambda candidate:
            0.0,
        selection_count=1,
    )

    generation.next()

    assert len(population) == 2