"""
tests/test_optimization_problem.py
"""

from __future__ import annotations

import random

from optimization.optimization_problem import (
    OptimizationProblem,
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


class DummyBuilder:

    def build(
        self,
        parameters,
    ):
        return parameters


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


def test_problem_returns_population():

    problem = OptimizationProblem(
        parameter_template=create_template(),
        builder=DummyBuilder(),
        simulator=lambda mechanism: mechanism,
        fitness=lambda _: 1.0,
        random_generator=random.Random(1),
    )

    result = problem.optimize(
        population_size=5,
        generations=2,
        children_per_generation=5,
        selection_count=2,
    )

    assert isinstance(
        result,
        Population,
    )


def test_problem_population_size():

    problem = OptimizationProblem(
        parameter_template=create_template(),
        builder=DummyBuilder(),
        simulator=lambda mechanism: mechanism,
        fitness=lambda _: 1.0,
        random_generator=random.Random(1),
    )

    result = problem.optimize(
        population_size=8,
        generations=1,
        children_per_generation=8,
    )

    assert len(result) == 8