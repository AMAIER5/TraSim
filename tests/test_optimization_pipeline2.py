"""
tests/test_optimization_pipeline.py
"""

from __future__ import annotations

import random

from optimization.optimization_pipeline import (
    OptimizationPipeline,
)

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
                minimum=10,
                maximum=100,
                value=50,
            ),
        )
    )


def test_pipeline_returns_population():

    pipeline = OptimizationPipeline(
        problem=OptimizationProblem(
            parameter_template=create_template(),
            builder=DummyBuilder(),
            simulator=lambda mechanism: mechanism,
            fitness=lambda mechanism: 0.0,
            random_generator=random.Random(1),
        )
    )

    result = pipeline.run(
        population_size=10,
        generations=2,
        children_per_generation=10,
    )

    assert isinstance(
        result,
        Population,
    )


def test_pipeline_population_size():

    pipeline = OptimizationPipeline(
        problem=OptimizationProblem(
            parameter_template=create_template(),
            builder=DummyBuilder(),
            simulator=lambda mechanism: mechanism,
            fitness=lambda mechanism: 0.0,
            random_generator=random.Random(1),
        )
    )

    result = pipeline.run(
        population_size=7,
        generations=1,
        children_per_generation=7,
    )

    assert len(result) == 7