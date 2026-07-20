"""
optimization/optimization_pipeline.py

End-to-end optimization pipeline.

This class connects

ParameterSet
    ->
Mechanism
    ->
Simulation
    ->
Fitness
    ->
Evolution.
"""

from __future__ import annotations

from optimization.optimization_problem import (
    OptimizationProblem,
)

from optimization.population import (
    Population,
)


class OptimizationPipeline:
    """
    Executes a complete optimization workflow.
    """

    def __init__(
        self,
        *,
        problem: OptimizationProblem,
    ):

        self.problem = problem

    def run(
        self,
        *,
        population_size: int,
        generations: int,
        children_per_generation: int,
    ) -> Population:
        """
        Execute optimization.
        """

        return self.problem.optimize(
            population_size=population_size,
            generations=generations,
            children_per_generation=children_per_generation,
        )