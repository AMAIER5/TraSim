"""
tests/test_multistage_curve_fitness.py
"""

from __future__ import annotations

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve
from simulation.simulation_result import SimulationResult


def test_curve_fitness_uses_last_stage_output():
    """
    Fitness must evaluate the transfer curve of the final stage,
    not the first stage.

    Fix #9: evaluate() now uses result.input_angles (the last
    stage's inputs) instead of input_result.input_angles.
    The last stage has inputs = (10, 11, 12) and outputs =
    (20, 21, 22).  Target = angle + 10 → (20, 21, 22).  MAE = 0.
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

    result = fitness.evaluate(
        simulation,
    )

    assert result == 0.0


def test_curve_fitness_does_not_use_first_stage_output():
    """
    A perfect second stage must remain perfect even if
    the first stage has an unrelated transfer curve.

    Fix #9: The last stage has inputs = (0, 1, 2) and
    outputs = (5, 6, 7).  Target = angle + 5 → (5, 6, 7).
    MAE = 0.

    Note: Both stages happen to share the same input grid
    (0, 1, 2) in this test, so the fix does not change the
    result here — but it confirms the last-stage output
    is used, not the first-stage output (100, 100, 100).
    """

    simulation = (
        SimulationResult(
            input_angles=(
                0.0,
                1.0,
                2.0,
            ),
            output_angles=(
                100.0,
                100.0,
                100.0,
            ),
            success=True,
        ),
        SimulationResult(
            input_angles=(
                0.0,
                1.0,
                2.0,
            ),
            output_angles=(
                5.0,
                6.0,
                7.0,
            ),
            success=True,
        ),
    )

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle + 5.0,
        ),
    )

    result = fitness.evaluate(
        simulation,
    )

    assert result == 0.0


def test_curve_fitness_different_input_grids_uses_last_stage():
    """
    Fix #9: When stages have different input grids, the
    target must be sampled at the last stage's input angles.

    stage1: inputs=(0, 1, 2), outputs=(10, 11, 12)
    stage2: inputs=(10, 11, 12), outputs=(20, 21, 22)

    With the fix, target is sampled at (10, 11, 12):
        target = (20, 21, 22), actual = (20, 21, 22) → MAE = 0.

    With the old code (sampling at first-stage inputs
    (0, 1, 2)):
        target = (10, 11, 12), actual = (20, 21, 22) → MAE = 10.
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


def test_curve_fitness_blocked_intermediate_returns_penalty():
    """
    Fix #9: If the last stage is blocked (success=False)
    with fewer points than the first stage, evaluate()
    must return a penalty, not raise a ValueError from
    TransferCurve (which would happen if first-stage
    input_angles were used with last-stage output_angles
    of different lengths).
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

    # Must be a penalty value, not an exception.
    assert result > 100.0


def test_curve_fitness_single_stage_works():
    """
    Fix #9: A single-stage simulation must still work
    correctly (result == simulation[-1] == simulation[0]).
    """

    simulation = (
        SimulationResult(
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
        ),
    )

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    result = fitness.evaluate(simulation)

    assert result == 0.0