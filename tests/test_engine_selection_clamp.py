"""tests/test_engine_selection_clamp.py

The selection count is clamped to the available candidates:
a population (or child count) smaller than the selection
count must not crash the run with "cannot select more
candidates than available" - GUI users can freely combine a
small population with a large selection count.
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
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import Reproduction


def _template() -> ParameterSet:
    return ParameterSet(
        parameters=(
            Parameter(
                name="x",
                minimum=-10.0,
                maximum=10.0,
                value=1.0,
            ),
        )
    )


def test_selection_larger_than_population_runs():
    rng = random.Random(0)
    template = _template()
    population = PopulationFactory(
        random_generator=rng,
        initial_spread=0.1,
    ).create(template, size=5)
    engine = EvolutionEngine(
        population=population,
        evaluator=lambda parameters: abs(
            parameters.get("x").value,
        ),
        selection_count=12,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                strength=0.1,
                random_generator=rng,
            ),
            random_generator=rng,
        ),
        max_generations=3,
    )
    engine.evaluate_population()
    generations = 0
    for _ in engine.run(children_count=3):
        generations += 1
        assert generations <= 3
    engine.close()
    assert generations == 3
    assert engine.best_score < float("inf")
