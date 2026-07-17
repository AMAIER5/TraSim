"""
optimization/evaluator.py

Evaluation interface for optimization candidates.
"""

from __future__ import annotations

from typing import Callable

from optimization.parameter_set import (
    ParameterSet,
)


class Evaluator:
    """
    Calculates a score for a parameter set.

    The evaluator only knows how to call
    the evaluation function.

    The actual simulation remains outside.
    """

    def __init__(
        self,
        *,
        evaluate_function: Callable[
            [ParameterSet],
            float,
        ],
    ):

        if not callable(
            evaluate_function
        ):

            raise TypeError(
                "evaluate_function must be callable"
            )

        self.evaluate_function = (
            evaluate_function
        )

    def evaluate(
        self,
        parameter_set: ParameterSet,
    ) -> float:
        """
        Return fitness score.
        """

        return self.evaluate_function(
            parameter_set
        )