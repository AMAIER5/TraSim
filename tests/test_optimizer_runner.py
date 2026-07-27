"""
tests/test_optimizer_runner.py

Tests for optimization runner.
"""

from __future__ import annotations

import random

from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.optimizer_runner import (
    OptimizerRunner,
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


def create_engine():

    return EvolutionEngine(
        population=Population(
            (
                create_candidate(20.0),
            )
        ),

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


def test_runner_executes_generations():

    runner = OptimizerRunner(
        engine=create_engine()
    )

    result = runner.run(
        generations=3,
        children_count=2,
    )

    assert len(result) == 2


def test_runner_updates_engine_population():

    engine = create_engine()

    runner = OptimizerRunner(
        engine=engine
    )

    result = runner.run(
        generations=1,
        children_count=2,
    )

    assert engine.population == result


def test_runner_rejects_invalid_generations():

    runner = OptimizerRunner(
        engine=create_engine()
    )

    try:

        runner.run(
            generations=0,
            children_count=1,
        )

        assert False

    except ValueError:

        assert True