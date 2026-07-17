"""
optimization/mechanism_optimizer.py

Adapter between mechanism simulation
and evolutionary optimization.
"""

from __future__ import annotations

from typing import Callable, Any

from optimization.parameter_set import (
    ParameterSet,
)


class MechanismOptimizer:
    """
    Evaluates mechanism candidates.

    Pipeline:

    ParameterSet
        ->
    Mechanism
        ->
    Simulation
        ->
    Fitness
    """

    def __init__(
        self,
        *,
        mechanism_factory: Callable[
            [ParameterSet],
            Any,
        ],
        simulator: Callable[
            [Any],
            Any,
        ],
        fitness: Callable[
            [Any],
            float,
        ],
    ):

        self.mechanism_factory = (
            mechanism_factory
        )

        self.simulator = simulator

        self.fitness = fitness

    def evaluate(
        self,
        parameters: ParameterSet,
    ) -> float:
        """
        Evaluate one mechanism candidate.
        """

        mechanism = (
            self.mechanism_factory(
                parameters
            )
        )

        result = (
            self.simulator(
                mechanism
            )
        )

        return self.fitness(
            result
        )