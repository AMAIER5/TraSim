"""
tests/test_optimization_problem.py
"""

from __future__ import annotations

import random

from analysis.curve_fitness import (
    CurveFitness,
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
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import (
    MotionRange,
)


class DummyBuilder:

    def build(
        self,
        parameters,
    ):
        return parameters


class DummySimulator(MechanismSimulator):

    def __init__(self):

        pass

    def simulate(
        self,
        mechanism,
    ):

        return mechanism


class DummyFitness(CurveFitness):

    def __init__(self):

        pass

    def evaluate(
        self,
        simulation,
    ) -> float:

        return 1.0


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


def create_motion():

    return MotionRange(
        start_angle=0.0,
        max_angle=1.0,
        step=0.5,
    )


def test_problem_returns_population():

    problem = OptimizationProblem(
        parameter_template=create_template(),
        builder=DummyBuilder(),
        simulator=DummySimulator(),
        fitness=DummyFitness(),
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
        simulator=DummySimulator(),
        fitness=DummyFitness(),
        random_generator=random.Random(1),
    )

    result = problem.optimize(
        population_size=8,
        generations=1,
        children_per_generation=8,
    )

    assert len(result) == 8