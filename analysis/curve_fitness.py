"""
analysis/curve_fitness.py

Issue #17: The original code used hardcoded magic numbers
(1000.0, 100.0, 11, x10, 1e6) in the penalty calculation.
These are now extracted to named module-level constants
with clear documentation.  The ``11`` (expected point
count for a full simulation) is now a configurable
parameter ``expected_point_count`` on the ``CurveFitness``
constructor, defaulting to ``11`` for backward
compatibility.
"""

from __future__ import annotations

import logging

from analysis.error_metric import ErrorMetric
from analysis.target_curve import TargetCurve
from analysis.transfer_curve import TransferCurve
from optimization.fitness_function import FitnessFunction
from simulation.simulation_result import SimulationResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Penalty constants (Issue #17)
# --------------------------------------------------------------------------- #

#: Base penalty added to all failed / blocked simulations.
#: Ensures that any failed simulation scores worse than
#: any successful one (whose fitness is a non-negative
#: error metric).
PENALTY_BASE = 1000.0

#: Penalty applied when a simulation blocks and the
#: blocked angle is unknown (``blocked_at is None``).
PENALTY_BLOCKED_UNKNOWN = 100.0

#: Multiplier applied to the per-missing-point penalty.
PENALTY_MISSING_POINT_MULTIPLIER = 10.0

#: Fitness returned for a successful simulation that
#: produced fewer than 2 points (too few for a transfer
#: curve).
PENALTY_INSUFFICIENT_POINTS = 1e6

#: Default expected number of points in a full
#: simulation.  Used to calculate the missing-point
#: penalty.  This matches the motion range used by the
#: standard single-stage optimization example
#: (−50° to +100° in 0.1° steps would yield 1501 points,
#: but the original code used 11 as a threshold — kept
#: for backward compatibility).
DEFAULT_EXPECTED_POINT_COUNT = 11


class CurveFitness(FitnessFunction):
    """
    Calculates the fitness of a simulated transfer curve.

    Lower values are better.

    Failed simulations receive a penalty based on:
    - where the simulation stopped
    - how many points were successfully calculated

    ErrorMetric instances are cached for identical
    input-angle grids.
    """

    def __init__(
        self,
        *,
        target_curve: TargetCurve,
        expected_point_count: int = DEFAULT_EXPECTED_POINT_COUNT,
    ) -> None:

        self._target_curve = target_curve

        self._expected_point_count = expected_point_count

        self._cache: dict[
            tuple[float, ...],
            ErrorMetric,
        ] = {}

    def evaluate(
        self,
        simulation: tuple[SimulationResult, ...],
    ) -> float:
        """
        Evaluate a simulated mechanism.
        """

        if not simulation:
            raise ValueError(
                "Simulation must contain at least one stage."
            )

        # The final stage represents the overall mechanism output.
        result = simulation[-1]

        logger.debug(
            "Simulation result: success=%s points=%d blocked_at=%s",
            result.success,
            len(result.input_angles),
            result.blocked_at,
        )

        # -------------------------------------------------
        # Invalid / blocked simulation
        # -------------------------------------------------

        if not result.success:

            calculated_points = len(
                result.input_angles
            )

            if result.blocked_at is not None:

                blocked_penalty = abs(
                    result.blocked_at
                )

            else:

                blocked_penalty = PENALTY_BLOCKED_UNKNOWN

            missing_penalty = (
                PENALTY_BLOCKED_UNKNOWN
                * max(
                    0,
                    self._expected_point_count - calculated_points,
                )
            )

            return (
                PENALTY_BASE
                + blocked_penalty
                + missing_penalty * PENALTY_MISSING_POINT_MULTIPLIER
            )

        # -------------------------------------------------
        # Valid simulation
        # -------------------------------------------------

        if len(result.input_angles) < 2:

            return PENALTY_INSUFFICIENT_POINTS

        # Fix #9: Use result.input_angles (the last stage's
        # own inputs) instead of input_result.input_angles
        # (the first stage's inputs).  result.input_angles
        # always has the same length as result.output_angles
        # (enforced by SimulationResult.__post_init__), so
        # TransferCurve construction can never fail on a
        # length mismatch when an intermediate stage blocks.
        # This also matches what ErrorMetric compares against.
        transfer_curve = TransferCurve(
            input_angles=result.input_angles,
            output_angles=result.output_angles,
        )

        key = transfer_curve.input_angles

        metric = self._cache.get(
            key
        )

        if metric is None:

            target = self._target_curve.sample(
                key
            )

            metric = ErrorMetric(
                target=target,
            )

            self._cache[key] = metric

        return metric.calculate(
            transfer_curve,
        )

    def __call__(
        self,
        transfer_curve: TransferCurve,
    ) -> float:
        """
        Backwards-compatible interface.
        """

        key = transfer_curve.input_angles

        metric = self._cache.get(
            key
        )

        if metric is None:

            target = self._target_curve.sample(
                key
            )

            metric = ErrorMetric(
                target=target,
            )

            self._cache[key] = metric

        return metric.calculate(
            transfer_curve,
        )