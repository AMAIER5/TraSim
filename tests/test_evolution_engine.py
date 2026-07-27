"""
tests/test_evolution_engine.py

Tests for evolutionary optimization engine.
"""

from __future__ import annotations

import random

from optimization.evolution_engine import (
    EvolutionEngine,
)
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


def create_engine(
    population,
):

    return EvolutionEngine(
        population=population,
        evaluator=lambda candidate:
            candidate.get(
                "length"
            ).value,
        selection_count=1,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                random_generator=random.Random(1)
            )
        ),
    )


def test_engine_creates_next_generation():

    engine = create_engine(
        Population(
            (
                create_candidate(50.0),
                create_candidate(20.0),
            )
        )
    )

    result = engine.step(
        children_count=3,
    )

    assert len(result) == 3


def test_engine_keeps_best_candidate():

    engine = create_engine(
        Population(
            (
                create_candidate(50.0),
                create_candidate(20.0),
            )
        )
    )

    result = engine.step(
        children_count=1,
    )

    assert result[0].get(
        "length"
    ).value != 50.0


def test_engine_updates_population():

    population = Population(
        (
            create_candidate(30.0),
        )
    )

    engine = create_engine(
        population
    )

    result = engine.step(
        children_count=2,
    )

    assert engine.population == result