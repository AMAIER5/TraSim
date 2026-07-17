"""
tests/test_mechanism_optimizer.py

Tests for mechanism optimization adapter.
"""

from __future__ import annotations

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_set import (
    ParameterSet,
)


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


def test_optimizer_calls_mechanism_factory():

    called = False

    def factory(parameters):

        nonlocal called

        called = True

        return "mechanism"


    optimizer = MechanismOptimizer(
        mechanism_factory=factory,
        simulator=lambda mechanism:
            "curve",
        fitness=lambda curve:
            1.0,
    )

    result = optimizer.evaluate(
        create_parameter_set()
    )

    assert called

    assert result == 1.0


def test_optimizer_returns_fitness_score():

    optimizer = MechanismOptimizer(
        mechanism_factory=lambda parameters:
            "mechanism",
        simulator=lambda mechanism:
            "curve",
        fitness=lambda curve:
            12.5,
    )

    result = optimizer.evaluate(
        create_parameter_set()
    )

    assert result == 12.5


def test_optimizer_does_not_modify_parameters():

    parameters = create_parameter_set()

    optimizer = MechanismOptimizer(
        mechanism_factory=lambda parameters:
            "mechanism",
        simulator=lambda mechanism:
            "curve",
        fitness=lambda curve:
            0.0,
    )

    optimizer.evaluate(
        parameters
    )

    assert parameters.get(
        "length"
    ).value == 50.0