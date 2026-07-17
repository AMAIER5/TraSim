"""
optimization/optimizer_runner.py

High level execution wrapper for optimization.
"""

from __future__ import annotations

from optimization.evolution_engine import (
    EvolutionEngine,
)

from optimization.population import (
    Population,
)


class OptimizerRunner:
    """
    Runs multiple evolutionary steps.
    """

    def __init__(
        self,
        *,
        engine: EvolutionEngine,
    ):

        self.engine = engine

    def run(
        self,
        *,
        generations: int,
        children_count: int,
    ) -> Population:
        """
        Execute optimization process.
        """

        if generations <= 0:

            raise ValueError(
                "generations must be positive"
            )

        population = (
            self.engine.population
        )

        for _ in range(generations):

            population = self.engine.step(
                children_count=children_count,
            )

        return population