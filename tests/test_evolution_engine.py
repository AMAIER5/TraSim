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


# ---------------------------------------------------------------------------
# Issue #5: update_best docstring and run() staleness
# ---------------------------------------------------------------------------

def test_update_best_returns_none():
    """
    Issue #5: update_best() returns None (the docstring
    previously claimed "Returns True").
    """

    engine = create_engine(
        Population(
            (
                create_candidate(50.0),
                create_candidate(20.0),
            )
        )
    )

    engine.evaluate_population()

    result = engine.update_best()

    assert result is None


def test_update_best_sets_best_candidate():

    engine = create_engine(
        Population(
            (
                create_candidate(50.0),
                create_candidate(20.0),
            )
        )
    )

    engine.evaluate_population()
    engine.update_best()

    assert engine.best_candidate is not None

    assert engine.best_score == 20.0


def test_update_best_resets_stagnation_on_improvement():

    engine = create_engine(
        Population(
            (
                create_candidate(50.0),
            )
        )
    )

    engine.evaluate_population()
    engine.update_best()

    assert engine._stagnation_counter == 0


def test_update_best_increments_stagnation_on_no_improvement():

    engine = create_engine(
        Population(
            (
                create_candidate(50.0),
            )
        )
    )

    # First update sets best_score = 50.
    engine.evaluate_population()
    engine.update_best()

    # Same population: no improvement.
    engine.evaluate_population()
    engine.update_best()

    assert engine._stagnation_counter == 1


def test_run_best_candidate_not_stale():
    """
    Issue #5: After run() finishes, the final population
    must have been evaluated.  The best_candidate is the
    overall best across all generations (it may come from
    an earlier generation if that was better), but the
    final generation's children must not be skipped.

    We verify this by checking that the evaluator was
    called on every member of the final population.
    """

    eval_calls = []

    def tracking_evaluator(candidate):
        eval_calls.append(candidate)
        return candidate.get("length").value

    initial = Population(
        (
            create_candidate(90.0),
            create_candidate(80.0),
        )
    )

    engine = EvolutionEngine(
        population=initial,
        evaluator=tracking_evaluator,
        selection_count=1,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                random_generator=random.Random(42),
            ),
        ),
        max_generations=3,
    )

    generations = list(
        engine.run(children_count=2)
    )

    assert len(generations) == 3
    assert engine.stop_reason == "max_generations_reached"

    # The final population must have been evaluated.
    final_population = engine.population
    final_evaluated = set(eval_calls[-len(final_population):])
    final_members = set(final_population.members)

    assert final_members == final_evaluated


def test_run_evaluates_initial_population():
    """
    Issue #5: run() must evaluate the initial population
    before yielding generation 0.
    """

    engine = create_engine(
        Population(
            (
                create_candidate(50.0),
                create_candidate(20.0),
            )
        )
    )

    # Before run, scores are empty.
    assert engine.scores == {}

    list(engine.run(children_count=1))

    # After the first iteration, scores must be populated.
    assert len(engine.scores) >= 1