"""
analysis/curve_fitness.py

Fitness function for mechanism optimization.

Blocking mechanisms receive FINITE penalties based on:
- How early the blocking occurs (closer to start = worse)
- How many points are missing
- A finer gradient: the partial curve error over the
  successfully simulated points, which guides the optimizer
  toward non-blocking configurations even when the block
  position and missing-point count are identical.

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
from analysis.target_curve import TargetCurve, DiscreteTargetCurve
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

# Weight of the partial-curve error term added to blocking penalties.
# The partial error is the mean absolute deviation of the successfully
# simulated output angles from the target curve, sampled at those same
# input angles.  Adding it gives the optimizer a smooth gradient inside
# the blocking region (where many candidates share an identical block
# position and point count) without disturbing the non-blocking regime,
# because the term is strictly additive on top of PENALTY_BASE.
PARTIAL_CURVE_WEIGHT = 50.0


def partial_curve_error(
    target: TargetCurve,
    input_angles: tuple[float, ...],
    output_angles: tuple[float, ...],
) -> float:
    """
    Mean absolute error of a partial output against the target.

    Used inside the blocking region to provide a finer gradient:
    among blocking candidates that block at the same position with
    the same number of points, the one whose partial output is closer
    to the target scores better.

    Fewer than two points cannot define a curve, so the error is
    zero (no gradient is available in that case).

    For a non-interpolating ``DiscreteTargetCurve`` only the input
    angles that coincide with a support point are compared; every
    other angle is ignored.  This keeps the comparison at the
    prescribed support points and never uses interpolation between
    them.
    """

    if len(input_angles) < 2 or len(output_angles) < 2:
        return 0.0

    if len(input_angles) != len(output_angles):
        return 0.0

    pairs = _matched_pairs(
        target,
        input_angles,
        output_angles,
    )

    if len(pairs) < 2:
        return 0.0

    total = 0.0
    for target_output, actual_output in pairs:
        total += abs(actual_output - target_output)

    return total / len(pairs)


def _matched_pairs(
    target: TargetCurve,
    input_angles: tuple[float, ...],
    output_angles: tuple[float, ...],
) -> list[tuple[float, float]]:
    """
    Pairs of (target_output, actual_output) used for comparison.

    For a regular ``TargetCurve`` (defined everywhere) every
    simulated input angle is compared, sampling the target at the
    simulated input angles exactly as before.

    For a ``DiscreteTargetCurve`` only the input angles that
    coincide (within tolerance) with a support point contribute;
    the target output is taken directly from that support point,
    so no interpolation between support points is ever used.
    """

    if isinstance(target, DiscreteTargetCurve):

        pairs: list[tuple[float, float]] = []

        for input_angle, actual_output in zip(
            input_angles,
            output_angles,
        ):
            index = target.support_index(input_angle)

            if index is None:
                continue

            pairs.append(
                (
                    target.output_angles[index],
                    actual_output,
                )
            )

        return pairs

    target_curve = target.sample(input_angles)

    return [
        (target_output, actual_output)
        for target_output, actual_output in zip(
            target_curve.output_angles,
            output_angles,
        )
    ]


class CurveFitness(FitnessFunction):
    """
    Fitness with position-based blocking penalties and a finer
    gradient inside the blocking region.

    For multi-stage mechanisms: ALL stages must succeed.
    Blocking in ANY stage results in penalty.

    Penalty formula for blocking (with finer gradient):
        fitness = PENALTY_BASE +
                  (PENALTY_MAX_BLOCKING - PENALTY_BASE) * (1 - normalized_block_position) +
                  missing_points * PENALTY_MISSING_POINT +
                  PARTIAL_CURVE_WEIGHT * partial_curve_error

    Where normalized_block_position is the FIRST blocking position,
    normalized across the motion range:
        - 0.0 at motion start -> MAXIMUM penalty (PENALTY_MAX_BLOCKING)
        - 1.0 at motion end -> MINIMUM penalty (PENALTY_BASE)

    The partial-curve error term (``finer_gradient``) compares the
    successfully simulated output angles of the final stage against
    the target curve.  It is strictly additive, so it can only
    increase a blocking fitness; it never breaks the guarantee that
    blocking fitness >= PENALTY_BASE and non-blocking fitness
    < PENALTY_BASE.  Setting ``finer_gradient=False`` disables the
    term, collapsing the blocking penalty to the plain position +
    missing-point model.

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
        finer_gradient: bool = True,
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
        finer_gradient : bool
            When True (default), add a partial-curve error term to
            blocking penalties so the optimizer gets a smoother gradient
            inside the blocking region.  Set False for the plain
            position + missing-point penalty.
        """
        self._target_curve = target_curve
        self._motion_start = motion_start
        self._motion_range = motion_range
        self._finer_gradient = finer_gradient
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

            # Finer gradient: partial curve error over the successfully
            # simulated points of the final stage.  This rewards mechanisms
            # whose partial output is closer to the target, giving the
            # optimizer a smooth gradient inside the blocking region.  The
            # term is strictly additive, so it cannot pull a blocking
            # fitness below PENALTY_BASE.
            partial_term = 0.0
            if self._finer_gradient:
                final_result = simulation[-1]
                partial_error = partial_curve_error(
                    self._target_curve,
                    final_result.input_angles,
                    final_result.output_angles,
                )
                partial_term = PARTIAL_CURVE_WEIGHT * partial_error

            fitness = PENALTY_BASE + blocking_penalty + missing_penalty + partial_term

            logger.debug(
                "Blocking detected in stage: earliest_at=%s, points=%d, fitness=%s",
                earliest_block_angle, calculated_points, fitness
            )

            return fitness

        # --- ALL STAGES SUCCESSFUL: EVALUATE CURVE FIT ---
        # Only if ALL stages succeeded do we evaluate the curve fit.
        #
        # For a non-interpolating DiscreteTargetCurve the fitness is the
        # mean absolute error of the overall transfer curve: the original
        # support-point inputs (simulation[0].input_angles) are compared
        # against the final stage outputs (simulation[-1].output_angles),
        # and the target is sampled only at those support points.  No
        # interpolation between support points is used.  This keeps the
        # established single-stage behaviour (where simulation[0] ==
        # simulation[-1]) and extends it correctly to chained stages:
        # the support points flow through the kinematic chain as the
        # first-stage inputs, and the last-stage outputs are matched to
        # the target at those same support points.
        final_result = simulation[-1]

        if isinstance(self._target_curve, DiscreteTargetCurve):
            first_result = simulation[0]

            if len(first_result.input_angles) < 2:
                return PENALTY_INSUFFICIENT_POINTS

            pairs = _matched_pairs(
                self._target_curve,
                first_result.input_angles,
                final_result.output_angles,
            )

            if len(pairs) < 2:
                return PENALTY_INSUFFICIENT_POINTS

            total = 0.0
            for target_output, actual_output in pairs:
                total += abs(actual_output - target_output)

            curve_fitness = total / len(pairs)

            logger.debug(
                "Non-blocking discrete solution: fitness=%s",
                curve_fitness,
            )

            return curve_fitness

        result = final_result  # Final stage result (legacy path)

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

        For a non-interpolating ``DiscreteTargetCurve`` the mean
        absolute error is computed only at the transfer-curve input
        angles that coincide with a support point, without
        interpolating between support points.
        """
        if isinstance(self._target_curve, DiscreteTargetCurve):

            pairs = _matched_pairs(
                self._target_curve,
                transfer_curve.input_angles,
                transfer_curve.output_angles,
            )

            if len(pairs) < 2:
                return PENALTY_INSUFFICIENT_POINTS

            total = 0.0
            for target_output, actual_output in pairs:
                total += abs(actual_output - target_output)

            return total / len(pairs)

        key = transfer_curve.input_angles
        if key not in self._cache:
            self._cache[key] = ErrorMetric(target=self._target_curve.sample(key))

        return self._cache[key].calculate(transfer_curve)
