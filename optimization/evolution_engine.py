"""
optimization/evolution_engine.py

High level evolutionary optimization loop.
"""

from __future__ import annotations

from typing import Callable

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.population import (
    Population,
)

from optimization.reproduction import (
    Reproduction,
)

from optimization.selection import (
    Selection,
)


class EvolutionEngine:
    """
    Executes evolutionary optimization steps.

    The engine coordinates:
    
    - evaluation
    - selection
    - reproduction
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
        reproduction: Reproduction,
    ):

        self.population = population

        self.evaluator = evaluator

        self.selection_count = (
            selection_count
        )

        self.selection = Selection()

        self.reproduction = reproduction

    def step(
        self,
        *,
        children_count: int,
    ) -> Population:
        """
        Execute one evolutionary generation.
        """

        scores = {
            candidate:
                self.evaluator(candidate)
            for candidate
            in self.population
        }

        survivors = self.selection.select(
            self.population,
            scores,
            count=self.selection_count,
        )

        next_population = (
            self.reproduction.create(
                survivors,
                count=children_count,
            )
        )

        self.population = (
            next_population
        )

        return next_population