"""
analysis/curve_fitness.py

Fitness function for mechanism optimization.

Blocking mechanisms receive FINITE penalties based on:
- How early the blocking occurs (closer to start = worse)
- How many points are missing

CRITICAL: For multi-stage mechanisms, ALL stages must succeed.
If ANY stage blocks, the entire mechanism receives a penalty.
This prevents the optimizer from accepting solutions where
intermediate stages block but later stages still produce output.

The key insight: simulation is a tuple of SimulationResult, one per stage.
We must check ALL of them for blocking, not just the final stage.
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
# These ensure blocking mechanisms always score worse than non-blocking ones
PENALTY_BASE = 10000.0           # Minimum penalty for any blocking
PENALTY_MAX_BLOCKING = 100000.0  # Maximum penalty (blocking at motion start)
PENALTY_MISSING_POINT = 100.0   # Per missing point
PENALTY_INSUFFICIENT_POINTS = 100000.0  # Too few points for valid curve


class CurveFitness(FitnessFunction):
    """
    Fitness with position-based blocking penalties.

    For multi-stage mechanisms: ALL stages must succeed.
    Blocking in ANY stage results in penalty.

    Penalty formula for blocking:
        fitness = PENALTY_BASE +
                  (PENALTY_MAX_BLOCKING - PENALTY_BASE) * (1 - normalized_block_position) +
                  missing_points * PENALTY_MISSING_POINT

    Where normalized_block_position is the FIRST blocking position,
    normalized across the motion range:
        - 0.0 at motion start -> MAXIMUM penalty (PENALTY_MAX_BLOCKING)
        - 1.0 at motion end -> MINIMUM penalty (PENALTY_BASE)

    Guarantees:
        - Non-blocking: fitness < PENALTY_BASE (10000)
        - Blocking: fitness >= PENALTY_BASE
        - Earlier blocking = higher fitness = worse
        - All blocking mechanisms can be ranked against each other
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
            Start of motion range in radians (for position-based penalty)
        motion_range : float
            Total motion range in radians (end - start, for normalization)
        """
        self._target_curve = target_curve
        self._motion_start = motion_start
        self._motion_range = motion_range
        self._cache: dict[tuple[float, ...], ErrorMetric] = {}

    def evaluate(self, simulation: tuple[SimulationResult, ...]) -> float:
        """
        Evaluate a simulated mechanism.

        The simulation parameter is a TUPLE of SimulationResult objects,
        one for each stage in the mechanism. For a multi-stage mechanism,
        ALL stages must succeed (not block) for the mechanism to be valid.

        Returns
        -------
        float
            Fitness value (lower is better)
            - Non-blocking (all stages succeed): < 10000
            - Blocking (any stage fails): >= 10000 (earlier blocking = higher)
        """
        if not simulation:
            raise ValueError("Simulation must contain at least one stage.")

        # --- CHECK ALL STAGES FOR BLOCKING ---
        # CRITICAL: We must check EVERY stage, not just the final one
        # If ANY stage blocks, the entire mechanism is invalid
        all_success = True
        earliest_block_angle = None
        min_points = float('inf')

        for result in simulation:
            if not result.success:
                all_success = False
                # Track the earliest blocking angle across all stages
                if result.blocked_at is not None:
                    if earliest_block_angle is None or result.blocked_at < earliest_block_angle:
                        earliest_block_angle = result.blocked_at
                # Track minimum points across all stages
                if len(result.input_angles) < min_points:
                    min_points = len(result.input_angles)

        # --- ANY STAGE BLOCKED: APPLY PENALTY ---
        if not all_success:
            calculated_points = int(min_points)

            # Position-based penalty (earlier = worse)
            if earliest_block_angle is not None and self._motion_range > 0:
                # Normalize block position: 0.0 at start, 1.0 at end
                normalized_pos = max(0.0, min(1.0,
                    (earliest_block_angle - self._motion_start) / self._motion_range))
                # Invert: blocking at start (pos=0) -> max penalty, at end (pos=1) -> min penalty
                blocking_penalty = (PENALTY_MAX_BLOCKING - PENALTY_BASE) * (1.0 - normalized_pos)
            else:
                # Unknown block position: use maximum penalty
                blocking_penalty = PENALTY_MAX_BLOCKING - PENALTY_BASE

            # Missing points penalty (encourages more complete simulations)
            expected_points = 11  # Standard for comparison
            missing_penalty = max(0, expected_points - calculated_points) * PENALTY_MISSING_POINT

            fitness = PENALTY_BASE + blocking_penalty + missing_penalty
            
            logger.debug(
                "Blocking detected in stage: earliest_at=%s, points=%d, fitness=%s",
                earliest_block_angle, calculated_points, fitness
            )
            
            return fitness

        # --- ALL STAGES SUCCESSFUL: EVALUATE CURVE FIT ---
        # Only if ALL stages succeeded do we evaluate the curve fit
        result = simulation[-1]  # Final stage result
        
        if len(result.input_angles) < 2:
            return PENALTY_INSUFFICIENT_POINTS

        transfer_curve = TransferCurve(
            input_angles=result.input_angles,
            output_angles=result.output_angles,
        )

        key = transfer_curve.input_angles
        if key not in self._cache:
            self._cache[key] = ErrorMetric(target=self._target_curve.sample(key))

        curve_fitness = self._cache[key].calculate(transfer_curve)
        
        logger.debug("Non-blocking solution: fitness=%s", curve_fitness)
        
        return curve_fitness

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
