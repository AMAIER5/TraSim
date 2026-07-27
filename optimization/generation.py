"""
optimization/generation.py

Evolutionary generation step.
"""

from __future__ import annotations

from collections.abc import Callable

from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population import (
    Population,
)


class Generation:
    """
    Executes one evolutionary step.

    Current strategy:

    1. evaluate candidates
    2. select best candidates
    3. return survivors

    Mutation will be added as a separate step.
    """

    def __init__(
        self,
        *,
        population: Population,
        evaluator: Callable[
            [ParameterSet],
            float,
        ],
        selection_count: int,
    ):

        self.population = population

        self.evaluator = evaluator

        self.selection_count = (
            selection_count
        )

    def next(
        self,
    ) -> Population:
        """
        Create next generation.
        """

        scores = {
            candidate:
                self.evaluator(candidate)
            for candidate
            in self.population
        }

        ranked = sorted(
            self.population,
            key=lambda candidate:
                scores[candidate],
        )

        return Population(
            tuple(
                ranked[
                    :self.selection_count
                ]
            )
        )