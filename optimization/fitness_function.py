"""
optimization/fitness_function.py

Protocol for mechanism fitness evaluation.
"""

from __future__ import annotations

from typing import Protocol

from simulation.simulation_result import SimulationResult
from validation.stage_validation_result import (
    StageValidationResult,
)


class FitnessFunction(Protocol):
    """
    Evaluates simulation results.

    Optional stage validation results may be supplied so the
    fitness can take mechanism validity (blocking over the full
    stage range) into account, not only the points sampled by
    the simulation.
    """

    def evaluate(
        self,
        simulation: tuple[SimulationResult, ...],
        validation: tuple[StageValidationResult, ...] | None = None,
    ) -> float:
        """
        Calculate the fitness value.

        Parameters
        ----------
        simulation:
            One SimulationResult per stage.
        validation:
            One StageValidationResult per stage, or None when
            no validation information is available.
        """
        ...