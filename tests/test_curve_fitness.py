"""
tests/test_curve_fitness.py

Issue #17: Added tests for named penalty constants and
the configurable ``expected_point_count`` parameter.
Also preserves all existing tests from the Issue #9 fix.
"""

from __future__ import annotations

from analysis.curve_fitness import (
    CurveFitness,
    PENALTY_BASE,
    PENALTY_BLOCKED_UNKNOWN,
    PENALTY_MISSING_POINT_MULTIPLIER,
    PENALTY_INSUFFICIENT_POINTS,
    DEFAULT_EXPECTED_POINT_COUNT,
)
from analysis.target_curve import (
    TargetCurve,
)
from analysis.transfer_curve import (
    TransferCurve,
)
from simulation.simulation_result import SimulationResult


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
# Issue #17: Named constants and expected_point_count
# ---------------------------------------------------------------------------

def test_penalty_constants_are_positive():

    assert PENALTY_BASE > 0
    assert PENALTY_BLOCKED_UNKNOWN > 0
    assert PENALTY_MISSING_POINT_MULTIPLIER > 0
    assert PENALTY_INSUFFICIENT_POINTS > 0
    assert DEFAULT_EXPECTED_POINT_COUNT > 0


def test_default_expected_point_count_is_11():
    """
    Issue #17: The default expected_point_count is 11,
    matching the original hardcoded magic number.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    assert fitness._expected_point_count == 11


def test_custom_expected_point_count():
    """
    Issue #17: expected_point_count is configurable.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        expected_point_count=1501,
    )

    assert fitness._expected_point_count == 1501


def test_penalty_uses_named_constants():
    """
    Issue #17: The penalty for a blocked simulation with
    known blocked_at must equal:
        PENALTY_BASE + |blocked_at|
            + PENALTY_BLOCKED_UNKNOWN
              * max(0, expected - points)
              * PENALTY_MISSING_POINT_MULTIPLIER
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        expected_point_count=11,
    )

    blocked = SimulationResult(
        input_angles=(0.0, 1.0),  # 2 points
        output_angles=(0.0, 1.0),
        success=False,
        blocked_at=3.0,
    )

    result = fitness.evaluate((blocked,))

    expected = (
        PENALTY_BASE
        + abs(3.0)
        + PENALTY_BLOCKED_UNKNOWN
        * max(0, 11 - 2)
        * PENALTY_MISSING_POINT_MULTIPLIER
    )

    assert result == expected


def test_penalty_blocked_at_none_uses_unknown_penalty():
    """
    Issue #17: When blocked_at is None, the blocked_penalty
    is PENALTY_BLOCKED_UNKNOWN.
    """

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        expected_point_count=11,
    )

    blocked = SimulationResult(
        input_angles=(0.0, 1.0),  # 2 points
        output_angles=(0.0, 1.0),
        success=False,
        blocked_at=None,
    )

    result = fitness.evaluate((blocked,))

    expected = (
        PENALTY_BASE
        + PENALTY_BLOCKED_UNKNOWN
        + PENALTY_BLOCKED_UNKNOWN
        * max(0, 11 - 2)
        * PENALTY_MISSING_POINT_MULTIPLIER
    )

    assert result == expected


def test_insufficient_points_returns_constant():
    """
    Issue #17: A successful simulation with < 2 points
    returns PENALTY_INSUFFICIENT_POINTS.
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


def test_custom_expected_count_affects_penalty():
    """
    Issue #17: A higher expected_point_count increases the
    missing-point penalty for the same blocked simulation.
    """

    blocked = SimulationResult(
        input_angles=(0.0, 1.0),  # 2 points
        output_angles=(0.0, 1.0),
        success=False,
        blocked_at=1.0,
    )

    fitness_low = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        expected_point_count=5,
    )

    fitness_high = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
        expected_point_count=50,
    )

    penalty_low = fitness_low.evaluate((blocked,))
    penalty_high = fitness_high.evaluate((blocked,))

    assert penalty_high > penalty_low