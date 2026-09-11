"""
tests/test_curve_fitness.py

Tests for CurveFitness, the position-based blocking penalty
and the finer gradient in the blocking region.

The fitness uses a two-tier model:

- Non-blocking (all stages succeed): fitness < PENALTY_BASE.
  Equal to the mean absolute error against the target curve.

- Blocking (any stage blocks): fitness >= PENALTY_BASE.
  A position-based penalty that is higher the earlier the
  mechanism blocks, plus a missing-point penalty, plus an
  optional partial-curve error that creates a finer gradient
  among blocking candidates so the optimizer is guided toward
  non-blocking configurations.

Earlier blocking is always penalised more heavily than later
blocking, and every blocking mechanism scores worse than
every non-blocking mechanism.
"""

from __future__ import annotations

from math import isclose

from analysis.curve_fitness import (
    CurveFitness,
    PENALTY_BASE,
    PENALTY_MAX_BLOCKING,
    PENALTY_MISSING_POINT,
    PENALTY_INSUFFICIENT_POINTS,
    PARTIAL_CURVE_WEIGHT,
    partial_curve_error,
)
from analysis.target_curve import TargetCurve
from analysis.transfer_curve import TransferCurve
from simulation.simulation_result import SimulationResult
from validation.stage_validation_result import (
    StageValidationResult,
)


def create_transfer_curve() -> TransferCurve:

    return TransferCurve(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            0.0,
            1.0,
            2.0,
        ),
    )


def test_identical_curve_has_zero_error():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    result = fitness(
        create_transfer_curve(),
    )

    assert result == 0.0


def test_shifted_curve_has_positive_error():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    simulated = TransferCurve(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            1.0,
            2.0,
            3.0,
        ),
    )

    assert fitness(simulated) > 0.0


def test_target_is_sampled_at_input_angles():

    sampled: dict[str, tuple[float, ...]] = {}

    class RecordingTargetCurve(TargetCurve):

        def sample(
            self,
            input_angles: tuple[float, ...],
        ) -> TransferCurve:

            sampled["angles"] = input_angles

            return super().sample(
                input_angles,
            )

    transfer = create_transfer_curve()

    fitness = CurveFitness(
        target_curve=RecordingTargetCurve(
            function=lambda angle: angle,
        ),
    )

    fitness(transfer)

    assert sampled["angles"] == transfer.input_angles


def test_metric_is_cached():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    curve = create_transfer_curve()

    fitness(curve)

    assert len(fitness._cache) == 1

    fitness(curve)

    assert len(fitness._cache) == 1


def test_evaluate_uses_last_stage_result():
    """
    The fitness of a two-stage simulation is determined
    by the last stage's transfer curve.

    Fix #9: evaluate() now uses result.input_angles (the
    last stage's own inputs) rather than input_result.
    input_angles (the first stage's inputs).  In this test
    both stages have identical input grids so the result
    is unchanged.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    stage1 = SimulationResult(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            10.0,
            11.0,
            12.0,
        ),
        success=True,
    )

    stage2 = SimulationResult(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            0.0,
            1.0,
            2.0,
        ),
        success=True,
    )

    result = fitness.evaluate(
        (
            stage1,
            stage2,
        )
    )

    assert result == 0.0


def test_uses_last_stage_output():
    """
    Fix #9: When stage1 and stage2 have different input
    grids, the target must be sampled at the last stage's
    input angles, not the first stage's.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    stage1 = SimulationResult(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            10.0,
            11.0,
        ),
        success=True,
    )

    stage2 = SimulationResult(
        input_angles=(
            10.0,
            11.0,
        ),
        output_angles=(
            10.0,
            11.0,
        ),
        success=True,
    )

    result = fitness.evaluate(
        (
            stage1,
            stage2,
        )
    )

    # Target sampled at (10, 11): target = (10, 11).
    # Actual outputs: (10, 11).  MAE = 0.
    assert result == 0.0


def test_evaluate_blocked_simulation_returns_penalty():
    """
    Fix #9: A blocked simulation (success=False) must return
    a penalty value, not raise an exception.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    blocked = SimulationResult(
        input_angles=(0.0, 1.0),
        output_angles=(0.0, 1.0),
        success=False,
        blocked_at=2.0,
    )

    result = fitness.evaluate(
        (blocked,),
    )

    assert result > 100.0


def test_evaluate_blocked_intermediate_stage_returns_penalty():
    """
    Fix #9: If an intermediate stage blocks, the last stage
    has fewer points than the first.  evaluate() must use
    result.input_angles (which always matches
    result.output_angles in length) and return a penalty,
    not crash with a ValueError from TransferCurve.
    """

    simulation = (
        SimulationResult(
            input_angles=(
                0.0,
                1.0,
                2.0,
                3.0,
            ),
            output_angles=(
                10.0,
                11.0,
                12.0,
                13.0,
            ),
            success=True,
        ),
        # Second stage blocks after 2 points.
        SimulationResult(
            input_angles=(
                10.0,
                11.0,
            ),
            output_angles=(
                20.0,
                21.0,
            ),
            success=False,
            blocked_at=12.0,
        ),
    )

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle + 10.0,
        ),
    )

    result = fitness.evaluate(simulation)

    assert result > 100.0


def test_evaluate_uses_last_stage_input_angles_not_first():
    """
    Fix #9: When stages have different input grids, the
    target must be sampled at the last stage's input angles.

    stage2 inputs = (10, 11, 12), outputs = (20, 21, 22).
    Target = angle + 10 -> (20, 21, 22).  MAE = 0.

    If the old code (input_result.input_angles = (0, 1, 2))
    were used, the target would be sampled at (0, 1, 2) ->
    (10, 11, 12), and MAE would be ~10, not 0.
    """

    simulation = (
        SimulationResult(
            input_angles=(
                0.0,
                1.0,
                2.0,
            ),
            output_angles=(
                10.0,
                11.0,
                12.0,
            ),
            success=True,
        ),
        SimulationResult(
            input_angles=(
                10.0,
                11.0,
                12.0,
            ),
            output_angles=(
                20.0,
                21.0,
                22.0,
            ),
            success=True,
        ),
    )

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle + 10.0,
        ),
    )

    result = fitness.evaluate(simulation)

    assert result == 0.0


# ---------------------------------------------------------------------------
# Penalty constants
# ---------------------------------------------------------------------------

def test_penalty_constants_are_positive():

    assert PENALTY_BASE > 0
    assert PENALTY_MAX_BLOCKING > 0
    assert PENALTY_MISSING_POINT > 0
    assert PENALTY_INSUFFICIENT_POINTS > 0
    assert PARTIAL_CURVE_WEIGHT > 0


def test_blocking_fitness_is_at_least_penalty_base():
    """
    Every blocking mechanism must score at least PENALTY_BASE,
    so non-blocking solutions (fitness < PENALTY_BASE) always
    rank better.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    blocked = SimulationResult(
        input_angles=(0.0, 1.0),
        output_angles=(0.0, 1.0),
        success=False,
        blocked_at=9.0,
    )

    assert fitness.evaluate((blocked,)) >= PENALTY_BASE


def test_earlier_blocking_is_penalised_more():
    """
    Blocking earlier in the motion range must produce a
    higher fitness than blocking later.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    early = SimulationResult(
        input_angles=(0.0,),
        output_angles=(0.0,),
        success=False,
        blocked_at=1.0,
    )

    late = SimulationResult(
        input_angles=(0.0,),
        output_angles=(0.0,),
        success=False,
        blocked_at=9.0,
    )

    # Same number of points, so the only difference is the
    # block position -> early blocking is worse (higher).
    assert fitness.evaluate((early,)) > fitness.evaluate((late,))


def test_unknown_block_position_uses_maximum_penalty():
    """
    When blocked_at is None the block position is unknown,
    so the maximum position-based penalty is applied.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    unknown = SimulationResult(
        input_angles=(0.0,),
        output_angles=(0.0,),
        success=False,
        blocked_at=None,
    )

    assert fitness.evaluate((unknown,)) >= PENALTY_MAX_BLOCKING


def test_missing_points_increase_penalty():
    """
    A blocking simulation that produces fewer valid points
    must score worse than one producing more, at the same
    block position.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    few = SimulationResult(
        input_angles=(0.0,),
        output_angles=(0.0,),
        success=False,
        blocked_at=5.0,
    )

    many = SimulationResult(
        input_angles=(0.0, 1.0, 2.0, 3.0),
        output_angles=(0.0, 1.0, 2.0, 3.0),
        success=False,
        blocked_at=5.0,
    )

    assert fitness.evaluate((few,)) > fitness.evaluate((many,))


def test_insufficient_points_returns_constant():
    """
    A successful simulation with fewer than 2 points
    returns PENALTY_INSUFFICIENT_POINTS (too few for a
    transfer curve).
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    single_point = SimulationResult(
        input_angles=(0.0,),
        output_angles=(0.0,),
        success=True,
    )

    result = fitness.evaluate((single_point,))

    assert result == PENALTY_INSUFFICIENT_POINTS


# ---------------------------------------------------------------------------
# Finer gradient in the blocking region
# ---------------------------------------------------------------------------

def test_partial_curve_error_zero_for_perfect_curve():
    """
    A partial output identical to the target produces a zero
    partial-curve error.
    """

    target = TargetCurve(function=lambda angle: angle + 10.0)

    error = partial_curve_error(
        target=target,
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(10.0, 11.0, 12.0),
    )

    assert error == 0.0


def test_partial_curve_error_positive_for_deviating_curve():
    """
    A partial output that deviates from the target produces a
    positive partial-curve error.
    """

    target = TargetCurve(function=lambda angle: angle + 10.0)

    error = partial_curve_error(
        target=target,
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(20.0, 21.0, 22.0),
    )

    assert error > 0.0


def test_partial_curve_error_requires_two_points():
    """
    Fewer than 2 points cannot define a curve, so the
    partial-curve error is zero (no gradient available).
    """

    target = TargetCurve(function=lambda angle: angle + 10.0)

    error = partial_curve_error(
        target=target,
        input_angles=(0.0,),
        output_angles=(5.0,),
    )

    assert error == 0.0


def test_finer_gradient_rewards_partial_curve_fit():
    """
    Two blocking mechanisms that block at the same position
    with the same number of points must still be ranked by
    how well their partial output matches the target.  The
    one closer to the target scores better (lower fitness).
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle + 10.0,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    good = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(10.0, 11.0, 12.0),
        success=False,
        blocked_at=5.0,
    )

    bad = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(20.0, 21.0, 22.0),
        success=False,
        blocked_at=5.0,
    )

    assert fitness.evaluate((good,)) < fitness.evaluate((bad,))


def test_finer_gradient_can_be_disabled():
    """
    With finer_gradient=False the blocking fitness collapses
    to the plain position + missing-point penalty, so two
    mechanisms with identical block position and point count
    score equally even if their partial curves differ.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle + 10.0,
        ),
        motion_start=0.0,
        motion_range=10.0,
        finer_gradient=False,
    )

    good = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(10.0, 11.0, 12.0),
        success=False,
        blocked_at=5.0,
    )

    bad = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(20.0, 21.0, 22.0),
        success=False,
        blocked_at=5.0,
    )

    assert fitness.evaluate((good,)) == fitness.evaluate((bad,))


def test_finer_gradient_keeps_blocking_above_base():
    """
    The partial-curve error term must never pull a blocking
    fitness below PENALTY_BASE, preserving the non-blocking
    / blocking separation.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    # A perfectly matching partial curve (error 0).
    good = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
        success=False,
        blocked_at=9.0,
    )

    assert fitness.evaluate((good,)) >= PENALTY_BASE


def test_finer_gradient_does_not_affect_non_blocking():
    """
    The finer gradient only applies to blocking simulations;
    a fully successful simulation is still scored by the
    plain curve error (< PENALTY_BASE).
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    ok = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
        success=True,
    )

    assert fitness.evaluate((ok,)) == 0.0
    assert fitness.evaluate((ok,)) < PENALTY_BASE


# ---------------------------------------------------------------------------
# Validation-driven blocking detection
#
# The simulation samples only the support points of the target curve, so a
# stage that blocks between support points can still report success.  When
# stage validation results are supplied, a stage whose validation reports
# ``valid=False`` is treated as blocking even if the simulation succeeded,
# closing the gap with information already computed at build time.
# ---------------------------------------------------------------------------

def test_validation_blocks_successful_simulation():
    """
    A simulation that reports success but whose validation
    reports an invalid stage must be treated as blocking and
    therefore score at least PENALTY_BASE.  This is the
    scenario from the reported example: the simulation never
    sampled the blocking angle (-46 deg) because it was not
    a support point of the target curve.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    simulation = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
        success=True,
    )

    validation = StageValidationResult(
        valid=False,
        checked_steps=50,
        failed_at_input_angle=4.0,
        reason="blocked",
        stage_id=0,
    )

    result = fitness.evaluate(
        (simulation,),
        (validation,),
    )

    assert result >= PENALTY_BASE


def test_validation_none_keeps_successful_simulation_below_base():
    """
    Without validation information, a successful simulation
    must remain a non-blocking result (< PENALTY_BASE).
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    simulation = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
        success=True,
    )

    assert fitness.evaluate((simulation,)) < PENALTY_BASE
    assert fitness.evaluate((simulation,), None) < PENALTY_BASE


def test_validation_block_position_drives_position_penalty():
    """
    When both the simulation and the validation report a
    block, the validation's failed_at_input_angle is the
    more precise position (sampled over the full stage range)
    and must drive the position-based penalty.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        motion_start=0.0,
        motion_range=10.0,
    )

    early_validation = StageValidationResult(
        valid=False,
        checked_steps=50,
        failed_at_input_angle=1.0,
        reason="blocked",
        stage_id=0,
    )
    late_validation = StageValidationResult(
        valid=False,
        checked_steps=50,
        failed_at_input_angle=9.0,
        reason="blocked",
        stage_id=0,
    )

    simulation = SimulationResult(
        input_angles=(0.0,),
        output_angles=(0.0,),
        success=True,
    )

    early = fitness.evaluate(
        (simulation,),
        (early_validation,),
    )
    late = fitness.evaluate(
        (simulation,),
        (late_validation,),
    )

    assert early > late


def test_validation_valid_does_not_block_successful_simulation():
    """
    A valid validation result must not turn a successful
    simulation into a blocking one.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    simulation = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
        success=True,
    )

    validation = StageValidationResult(
        valid=True,
        checked_steps=50,
        stage_id=0,
    )

    assert fitness.evaluate(
        (simulation,),
        (validation,),
    ) == 0.0
