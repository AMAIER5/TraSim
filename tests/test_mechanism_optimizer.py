"""
tests/test_mechanism_optimizer.py

Tests for mechanism optimization adapter.
"""

from __future__ import annotations

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

    def __init__(self):

        self.called = False

    def build(
        self,
        parameters: ParameterSet,
    ):

        self.called = True

        return "mechanism"


class DummySimulator:

    def __init__(self):

        self.called = False

    def simulate(
        self,
        mechanism,
    ):

        self.called = True

        return "simulation"


class DummyFitness(FitnessFunction):

    def __init__(
        self,
        value: float,
    ) -> None:

        self.called = False

        self.value = value

    def evaluate(
        self,
        simulation,
    ) -> float:

        self.called = True

        return self.value


def create_parameter_set():

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


def test_optimizer_calls_all_components():

    builder = DummyBuilder()

    simulator = DummySimulator()

    fitness = DummyFitness(
        value=1.0,
    )

    optimizer = MechanismOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
    )

    result = optimizer.evaluate(
        create_parameter_set(),
    )

    assert builder.called
    assert simulator.called
    assert fitness.called
    assert result == 1.0


def test_optimizer_returns_fitness_score():

    optimizer = MechanismOptimizer(
        builder=DummyBuilder(),
        simulator=DummySimulator(),
        fitness=DummyFitness(
            value=12.5,
        ),
    )

    result = optimizer.evaluate(
        create_parameter_set(),
    )

    assert result == 12.5


def test_optimizer_does_not_modify_parameters():

    parameters = create_parameter_set()

    optimizer = MechanismOptimizer(
        builder=DummyBuilder(),
        simulator=DummySimulator(),
        fitness=DummyFitness(
            value=0.0,
        ),
    )

    optimizer.evaluate(
        parameters,
    )

    assert (
        parameters.get(
            "length",
        ).value
        == 50.0
    )