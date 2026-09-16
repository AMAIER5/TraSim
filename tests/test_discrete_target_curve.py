"""
tests/test_discrete_target_curve.py

Tests for the non-interpolating target curve (DiscreteTargetCurve)
and the strict factories TargetCurve.from_points_strict /
TargetCurve.from_csv_strict.

A DiscreteTargetCurve is only defined at its support points.
Evaluating or sampling at an input angle that is not a support
point raises a ValueError, so the fitness is computed exclusively
at the prescribed support points and never uses interpolation
between them.
"""

from __future__ import annotations

import math

import pytest

from analysis.curve_fitness import (
    CurveFitness,
    partial_curve_error,
    PENALTY_BASE,
)
from analysis.target_curve import (
    DiscreteTargetCurve,
    TargetCurve,
)
from analysis.transfer_curve import TransferCurve
from simulation.simulation_result import SimulationResult


SUPPORT_INPUTS_DEG = (-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50)
SUPPORT_OUTPUTS_DEG = (60, 64, 68, 72, 76, 80, 86, 92, 98, 104, 110)


def support_inputs_rad() -> tuple[float, ...]:
    return tuple(
        math.radians(a) for a in SUPPORT_INPUTS_DEG
    )


def support_outputs_rad() -> tuple[float, ...]:
    return tuple(
        math.radians(a) for a in SUPPORT_OUTPUTS_DEG
    )


def make_discrete_target() -> DiscreteTargetCurve:
    return TargetCurve.from_points_strict(
        support_inputs_rad(),
        support_outputs_rad(),
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_from_points_strict_returns_discrete_target_curve():

    curve = make_discrete_target()

    assert isinstance(curve, DiscreteTargetCurve)


def test_from_points_strict_requires_same_length():

    with pytest.raises(ValueError):
        TargetCurve.from_points_strict(
            (0.0, 1.0),
            (0.0,),
        )


def test_from_points_strict_requires_sorted_input():

    with pytest.raises(ValueError):
        TargetCurve.from_points_strict(
            (1.0, 0.0),
            (0.0, 1.0),
        )


def test_from_points_strict_requires_two_points():

    with pytest.raises(ValueError):
        TargetCurve.from_points_strict(
            (0.0,),
            (0.0,),
        )


# ---------------------------------------------------------------------------
# evaluate / sample behaviour: only at support points
# ---------------------------------------------------------------------------

def test_evaluate_at_support_point_returns_output():

    curve = make_discrete_target()

    assert math.isclose(
        curve.evaluate(math.radians(-50.0)),
        math.radians(60.0),
    )

    assert math.isclose(
        curve.evaluate(math.radians(50.0)),
        math.radians(110.0),
    )


def test_evaluate_between_support_points_raises():

    curve = make_discrete_target()

    with pytest.raises(ValueError):
        curve.evaluate(math.radians(-45.0))


def test_sample_at_support_points_returns_matching_outputs():

    curve = make_discrete_target()

    inputs = support_inputs_rad()

    sampled = curve.sample(inputs)

    assert sampled.input_angles == inputs
    assert sampled.output_angles == support_outputs_rad()


def test_sample_between_support_points_raises():

    curve = make_discrete_target()

    with pytest.raises(ValueError):
        curve.sample((math.radians(-45.0),))


def test_support_index_returns_none_for_non_support():

    curve = make_discrete_target()

    assert curve.support_index(math.radians(-50.0)) == 0
    assert curve.support_index(math.radians(-45.0)) is None


# ---------------------------------------------------------------------------
# from_csv_strict
# ---------------------------------------------------------------------------

def test_from_csv_strict_loads_support_points(tmp_path):

    csv_file = tmp_path / "curve.csv"
    lines = ["input_angle,output_angle"]
    for inp, out in zip(
        SUPPORT_INPUTS_DEG,
        SUPPORT_OUTPUTS_DEG,
    ):
        lines.append(f"{inp},{out}")
    csv_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    curve = TargetCurve.from_csv_strict(csv_file)

    assert isinstance(curve, DiscreteTargetCurve)

    assert math.isclose(
        curve.evaluate(math.radians(-50.0)),
        math.radians(60.0),
    )

    with pytest.raises(ValueError):
        curve.evaluate(math.radians(-45.0))


def test_from_csv_strict_requires_columns(tmp_path):

    csv_file = tmp_path / "curve.csv"
    csv_file.write_text(
        "x,y\n0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        TargetCurve.from_csv_strict(csv_file)


# ---------------------------------------------------------------------------
# Fitness evaluated only at support points
# ---------------------------------------------------------------------------

def test_fitness_zero_when_all_support_points_matched():
    """
    A simulation whose output matches the target at every support
    point has fitness 0 (no interpolation involved).
    """

    fitness = CurveFitness(
        target_curve=make_discrete_target(),
    )

    result = SimulationResult(
        input_angles=support_inputs_rad(),
        output_angles=support_outputs_rad(),
        success=True,
    )

    assert fitness.evaluate((result,)) == 0.0


def test_fitness_zero_when_all_support_points_matched_multistage():
    """
    For a chained mechanism, the overall transfer (first-stage
    support inputs -> last-stage outputs) matching the target at
    every support point has fitness 0.
    """

    fitness = CurveFitness(
        target_curve=make_discrete_target(),
    )

    stage1 = SimulationResult(
        input_angles=support_inputs_rad(),
        output_angles=support_inputs_rad(),
        success=True,
    )

    stage2 = SimulationResult(
        input_angles=support_inputs_rad(),
        output_angles=support_outputs_rad(),
        success=True,
    )

    assert fitness.evaluate((stage1, stage2)) == 0.0


def test_fitness_punishes_only_a_deviation_at_a_support_point():
    """
    A simulated curve that deviates from the target at exactly one
    support point is punished only at that point.  The mean absolute
    error equals the deviation divided by the number of support
    points, confirming no interpolation between support points is
    used.
    """

    fitness = CurveFitness(
        target_curve=make_discrete_target(),
    )

    outputs = list(support_outputs_rad())
    deviation = math.radians(5.0)
    outputs[3] += deviation  # deviate at the 4th support point

    result = SimulationResult(
        input_angles=support_inputs_rad(),
        output_angles=tuple(outputs),
        success=True,
    )

    expected = deviation / len(SUPPORT_INPUTS_DEG)

    assert math.isclose(
        fitness.evaluate((result,)),
        expected,
        rel_tol=1e-12,
    )


def test_fitness_call_punishes_only_a_deviation_at_a_support_point():

    fitness = CurveFitness(
        target_curve=make_discrete_target(),
    )

    outputs = list(support_outputs_rad())
    deviation = math.radians(4.0)
    outputs[5] += deviation

    transfer = TransferCurve(
        input_angles=support_inputs_rad(),
        output_angles=tuple(outputs),
    )

    expected = deviation / len(SUPPORT_INPUTS_DEG)

    assert math.isclose(
        fitness(transfer),
        expected,
        rel_tol=1e-12,
    )


def test_input_angle_outside_support_points_is_not_evaluated():
    """
    A transfer curve that only contains an input angle outside the
    support points yields no comparable support points, so the
    backwards-compatible interface returns the insufficient-points
    penalty (the off-support point is ignored, not interpolated).
    """

    fitness = CurveFitness(
        target_curve=make_discrete_target(),
    )

    transfer = TransferCurve(
        input_angles=(math.radians(-45.0), math.radians(-35.0)),
        output_angles=(math.radians(1.0), math.radians(2.0)),
    )

    from analysis.curve_fitness import PENALTY_INSUFFICIENT_POINTS

    assert fitness(transfer) == PENALTY_INSUFFICIENT_POINTS


def test_partial_curve_error_ignores_non_support_points():
    """
    partial_curve_error compares only the input angles that
    coincide with a support point; off-support angles are ignored
    and never interpolated.
    """

    target = make_discrete_target()

    inputs = (
        math.radians(-50.0),
        math.radians(-45.0),
        math.radians(-40.0),
    )

    outputs = (
        math.radians(60.0),
        math.radians(999.0),
        math.radians(64.0),
    )

    error = partial_curve_error(
        target=target,
        input_angles=inputs,
        output_angles=outputs,
    )

    assert error == 0.0


def test_non_blocking_discrete_fitness_below_penalty_base():
    """
    A fully successful (non-blocking) simulation with a discrete
    target stays below PENALTY_BASE, preserving the blocking /
    non-blocking separation.
    """

    fitness = CurveFitness(
        target_curve=make_discrete_target(),
        motion_start=math.radians(-50.0),
        motion_range=math.radians(100.0),
    )

    result = SimulationResult(
        input_angles=support_inputs_rad(),
        output_angles=support_outputs_rad(),
        success=True,
    )

    assert fitness.evaluate((result,)) < PENALTY_BASE


# ---------------------------------------------------------------------------
# Per-support-point weighting
# ---------------------------------------------------------------------------

def test_weights_default_to_one():
    """
    A DiscreteTargetCurve constructed without weights gives every
    support point a weight of 1.
    """
    curve = make_discrete_target()
    assert curve.weights == tuple(
        1.0 for _ in SUPPORT_INPUTS_DEG
    )


def test_weights_are_stored():
    weights = tuple(
        float(i) for i in range(len(SUPPORT_INPUTS_DEG))
    )
    curve = TargetCurve.from_points_strict(
        support_inputs_rad(),
        support_outputs_rad(),
        weights=weights,
    )
    assert curve.weights == weights


def test_weights_length_must_match():
    with pytest.raises(ValueError):
        TargetCurve.from_points_strict(
            support_inputs_rad(),
            support_outputs_rad(),
            weights=(1.0, 2.0),
        )


def test_weights_must_be_non_negative():
    weights = tuple(
        -1.0 for _ in SUPPORT_INPUTS_DEG
    )
    with pytest.raises(ValueError):
        TargetCurve.from_points_strict(
            support_inputs_rad(),
            support_outputs_rad(),
            weights=weights,
        )


def test_weights_clamped_to_max():
    from analysis.target_curve import MAX_WEIGHT

    weights = tuple(
        1000.0 for _ in SUPPORT_INPUTS_DEG
    )
    curve = TargetCurve.from_points_strict(
        support_inputs_rad(),
        support_outputs_rad(),
        weights=weights,
    )
    assert curve.weights == tuple(
        MAX_WEIGHT for _ in SUPPORT_INPUTS_DEG
    )


def test_from_csv_strict_reads_optional_weight_column(tmp_path):
    csv_file = tmp_path / "curve.csv"
    lines = ["input_angle,output_angle,weight"]
    for inp, out in zip(
        SUPPORT_INPUTS_DEG,
        SUPPORT_OUTPUTS_DEG,
    ):
        lines.append(f"{inp},{out},2")
    csv_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    curve = TargetCurve.from_csv_strict(csv_file)
    assert isinstance(curve, DiscreteTargetCurve)
    assert curve.weights == tuple(
        2.0 for _ in SUPPORT_INPUTS_DEG
    )


def test_from_csv_strict_defaults_weight_to_one_without_column(tmp_path):
    csv_file = tmp_path / "curve.csv"
    lines = ["input_angle,output_angle"]
    for inp, out in zip(
        SUPPORT_INPUTS_DEG,
        SUPPORT_OUTPUTS_DEG,
    ):
        lines.append(f"{inp},{out}")
    csv_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    curve = TargetCurve.from_csv_strict(csv_file)
    assert curve.weights == tuple(
        1.0 for _ in SUPPORT_INPUTS_DEG
    )


def test_weighted_fitness_is_weighted_mean_absolute_error():
    """
    With a single high-weight support point the weighted mean
    absolute error equals the deviation at that point.
    """
    weights = [1.0] * len(SUPPORT_INPUTS_DEG)
    weights[3] = 10.0  # high weight at the 4th support point
    curve = TargetCurve.from_points_strict(
        support_inputs_rad(),
        support_outputs_rad(),
        weights=tuple(weights),
    )
    fitness = CurveFitness(target_curve=curve)
    outputs = list(support_outputs_rad())
    deviation = math.radians(5.0)
    outputs[3] += deviation
    result = SimulationResult(
        input_angles=support_inputs_rad(),
        output_angles=tuple(outputs),
        success=True,
    )
    weight_total = sum(weights)
    expected = (10.0 * deviation) / weight_total
    assert math.isclose(
        fitness.evaluate((result,)),
        expected,
        rel_tol=1e-12,
    )


def test_weighted_fitness_call_is_weighted_mean_absolute_error():
    weights = [1.0] * len(SUPPORT_INPUTS_DEG)
    weights[5] = 4.0
    curve = TargetCurve.from_points_strict(
        support_inputs_rad(),
        support_outputs_rad(),
        weights=tuple(weights),
    )
    fitness = CurveFitness(target_curve=curve)
    outputs = list(support_outputs_rad())
    deviation = math.radians(4.0)
    outputs[5] += deviation
    transfer = TransferCurve(
        input_angles=support_inputs_rad(),
        output_angles=tuple(outputs),
    )
    weight_total = sum(weights)
    expected = (4.0 * deviation) / weight_total
    assert math.isclose(
        fitness(transfer),
        expected,
        rel_tol=1e-12,
    )


def test_weighted_partial_curve_error_is_weighted():
    weights = [1.0] * len(SUPPORT_INPUTS_DEG)
    weights[0] = 3.0
    weights[2] = 2.0
    target = TargetCurve.from_points_strict(
        support_inputs_rad(),
        support_outputs_rad(),
        weights=tuple(weights),
    )
    inputs = (
        math.radians(-50.0),
        math.radians(-30.0),
    )
    outputs = (
        math.radians(61.0),  # 1deg off at support 0 (weight 3)
        math.radians(70.0),  # 2deg off at support 2 (weight 2)
    )
    error = partial_curve_error(
        target=target,
        input_angles=inputs,
        output_angles=outputs,
    )
    weight_total = 3.0 + 2.0
    expected = (
        3.0 * math.radians(1.0) + 2.0 * math.radians(2.0)
    ) / weight_total
    assert math.isclose(
        error,
        expected,
        rel_tol=1e-12,
    )


def test_zero_weight_support_point_is_ignored_in_fitness():
    """
    A support point with weight 0 does not contribute to the
    weighted mean absolute error.
    """
    weights = [1.0] * len(SUPPORT_INPUTS_DEG)
    weights[3] = 0.0
    curve = TargetCurve.from_points_strict(
        support_inputs_rad(),
        support_outputs_rad(),
        weights=tuple(weights),
    )
    fitness = CurveFitness(target_curve=curve)
    outputs = list(support_outputs_rad())
    deviation = math.radians(5.0)
    outputs[3] += deviation  # large deviation, but weight 0
    result = SimulationResult(
        input_angles=support_inputs_rad(),
        output_angles=tuple(outputs),
        success=True,
    )
    assert math.isclose(
        fitness.evaluate((result,)),
        0.0,
        abs_tol=1e-12,
    )
