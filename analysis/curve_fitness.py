"""
analysis/curve_fitness.py

Fitness function for mechanism optimization.

Blocking mechanisms receive FINITE penalties based on:
- How early the blocking occurs (closer to start = worse)
- How many points are missing
Non-blocking solutions always score better than blocking ones.
"""

from __future__ import annotations

import logging

from analysis.error_metric import ErrorMetric
from analysis.target_curve import TargetCurve
from analysis.transfer_curve import TransferCurve
from optimization.fitness_function import FitnessFunction
from simulation.simulation_result import SimulationResult

logger = logging.getLogger(__name__)

# Penalty constants - ALL FINITE
PENALTY_BASE = 10000.0           # Minimum for any blocking
PENALTY_MAX_BLOCKING = 100000.0  # Maximum for blocking at very start
PENALTY_MISSING_POINT = 100.0   # Per missing point
PENALTY_INSUFFICIENT_POINTS = 100000.0  # Too few points


class CurveFitness(FitnessFunction):
    """
    Fitness with position-based blocking penalties.

    Formula for blocking:
        fitness = PENALTY_BASE +
                  (PENALTY_MAX_BLOCKING - PENALTY_BASE) * (1 - normalized_block_position) +
                  missing_points * PENALTY_MISSING_POINT

    Where normalized_block_position is:
        - 0.0 at motion start (MAXIMUM penalty)
        - 1.0 at motion end (MINIMUM penalty)

    Guarantees:
        - Non-blocking: fitness < PENALTY_BASE (10000)
        - Blocking: fitness >= PENALTY_BASE
        - Earlier blocking = higher fitness = worse
    """

    def __init__(
        self,
        *,
        target_curve: TargetCurve,
        motion_start: float = 0.0,
        motion_range: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        target_curve : TargetCurve
            Desired input/output relationship
        motion_start : float
            Start of motion range in radians
        motion_range : float
            Total motion range in radians (end - start)
        """
        self._target_curve = target_curve
        self._motion_start = motion_start
        self._motion_range = motion_range
        self._cache: dict[tuple[float, ...], ErrorMetric] = {}

    def evaluate(self, simulation: tuple[SimulationResult, ...]) -> float:
        """
        Evaluate a simulated mechanism.

        Returns
        -------
        float
            Fitness value (lower is better)
        """
        if not simulation:
            raise ValueError("Simulation must contain at least one stage.")

        result = simulation[-1]  # Final stage result

        logger.debug(
            "Simulation: success=%s points=%d blocked_at=%s",
            result.success, len(result.input_angles), result.blocked_at
        )

        # --- BLOCKED SIMULATION ---
        if not result.success:
            calculated_points = len(result.input_angles)

            # Position-based penalty (earlier = worse)
            if result.blocked_at is not None and self._motion_range > 0:
                normalized_pos = max(0.0, min(1.0,
                    (result.blocked_at - self._motion_start) / self._motion_range))
                blocking_penalty = (PENALTY_MAX_BLOCKING - PENALTY_BASE) * (1.0 - normalized_pos)
            else:
                blocking_penalty = PENALTY_MAX_BLOCKING - PENALTY_BASE

            # Missing points penalty
            missing_penalty = max(0, 11 - calculated_points) * PENALTY_MISSING_POINT

            return PENALTY_BASE + blocking_penalty + missing_penalty

        # --- VALID SIMULATION ---
        if len(result.input_angles) < 2:
            return PENALTY_INSUFFICIENT_POINTS

        transfer_curve = TransferCurve(
            input_angles=result.input_angles,
            output_angles=result.output_angles,
        )

        key = transfer_curve.input_angles
        if key not in self._cache:
            self._cache[key] = ErrorMetric(target=self._target_curve.sample(key))

        return self._cache[key].calculate(transfer_curve)

    def __call__(
        self,
        transfer_curve: TransferCurve,
    ) -> float:
        """
        Backwards-compatible interface.
        """
        key = transfer_curve.input_angles
        if key not in self._cache:
            self._cache[key] = ErrorMetric(target=self._target_curve.sample(key))

        return self._cache[key].calculate(transfer_curve)
