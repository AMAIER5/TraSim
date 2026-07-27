"""
optimization/fitness_function.py

Protocol for mechanism fitness evaluation.
"""

from __future__ import annotations

from typing import Protocol

from simulation.simulation_result import SimulationResult


class FitnessFunction(Protocol):
    """
    Evaluates simulation results.
    """

    def evaluate(
        self,
        simulation: tuple[SimulationResult, ...],
    ) -> float:
        """
        Calculate the fitness value.
        """
        ...