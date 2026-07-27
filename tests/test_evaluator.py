"""
tests/test_evaluator.py

Tests for MechanismOptimizer.
"""

from __future__ import annotations

from mechanics.mechanism import Mechanism

from optimization.fitness_function import (
    FitnessFunction,
)
from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)
from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)


class DummyBuilder:

    def build(
        self,
        parameters: ParameterSet,
    ) -> Mechanism:

        return Mechanism(stages=())


class DummySimulator:

    def __init__(self):

        self.called = False

    def simulate(
        self,
        mechanism: Mechanism,
    ):

        self.called = True

        return ("simulation",)


class DummyFitness(FitnessFunction):

    def __init__(self):

        self.called = False

    def evaluate(
        self,
        simulation,
    ) -> float:

        self.called = True

        return 42.0


def create_parameter_set() -> ParameterSet:

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


def test_evaluator_returns_fitness_value():

    optimizer = MechanismOptimizer(
        builder=DummyBuilder(),
        simulator=DummySimulator(),
        fitness=DummyFitness(),
    )

    result = optimizer.evaluate(
        create_parameter_set()
    )

    assert result == 42.0


def test_evaluator_calls_simulator_and_fitness():

    simulator = DummySimulator()
    fitness = DummyFitness()

    optimizer = MechanismOptimizer(
        builder=DummyBuilder(),
        simulator=simulator,
        fitness=fitness,
    )

    optimizer.evaluate(
        create_parameter_set()
    )

    assert simulator.called
    assert fitness.called