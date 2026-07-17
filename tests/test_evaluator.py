"""
tests/test_evaluator.py

Tests for optimization evaluator.
"""

from __future__ import annotations

import math

from optimization.evaluator import (
    Evaluator,
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


def test_evaluator_returns_fitness_value():

    evaluator = Evaluator(
        evaluate_function=lambda parameters: 1.5,
    )

    result = evaluator.evaluate(
        create_parameter_set()
    )

    assert math.isclose(
        result,
        1.5,
    )


def test_evaluator_calls_function():

    called = False

    def function(parameters):

        nonlocal called

        called = True

        return 0.0

    evaluator = Evaluator(
        evaluate_function=function,
    )

    evaluator.evaluate(
        create_parameter_set()
    )

    assert called


def test_evaluator_is_independent_of_parameter_content():

    evaluator = Evaluator(
        evaluate_function=lambda parameters: 42.0,
    )

    result = evaluator.evaluate(
        create_parameter_set()
    )

    assert result == 42.0