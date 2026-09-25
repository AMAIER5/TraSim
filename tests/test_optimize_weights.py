"""
tests/test_optimize_weights.py

Regression tests for the target-curve loading in
examples/optimize.py.

optimize.py must load the target curve with
``TargetCurve.from_csv_strict`` so that

1. the optional per-support-point ``weight`` column of the
   target-curve CSV is honoured by the fitness, and
2. the fitness of a multi-stage mechanism compares the
   FINAL stage outputs against the target at the ORIGINAL
   support point inputs instead of sampling an
   interpolating curve at the last stage's input angles
   (which are the previous stage's outputs and can lie far
   outside the target's input range).
"""

from __future__ import annotations

import math
from pathlib import Path

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve
from examples.optimize import TARGET_FILE


def test_optimize_loads_weights_from_target_csv(tmp_path):
    """
    optimize.py's target curve must carry the weight column.

    A weighted support point must reach ``CurveFitness`` with
    its weight, not silently with weight 1.
    """

    csv_file = tmp_path / "target_curve.csv"
    csv_file.write_text(
        "input_angle,output_angle,weight\n0,10,1\n10,12,5\n20,14,1\n",
        encoding="utf-8",
    )

    curve = TargetCurve.from_csv_strict(csv_file)

    assert curve.weights == (1.0, 5.0, 1.0)


def test_optimize_target_file_loads_as_discrete_curve():
    """
    The example target file used by optimize.py must load as a
    non-interpolating curve that carries weights.
    """

    assert Path(TARGET_FILE).exists()

    curve = TargetCurve.from_csv_strict(TARGET_FILE)

    weights = curve.weights

    assert len(weights) == len(curve.input_angles)

    assert all(weight >= 1.0 for weight in weights)


def test_weighted_fitness_differs_from_unweighted(tmp_path):
    """
    The fitness of a fixed simulated curve must change when the
    weights change - proving the weights are actually applied.

    This exercises the exact evaluation path of optimize.py:
    a two-stage simulation tuple where the second stage's input
    angles lie OUTSIDE the target curve's input range (as with
    chained levers).  A strict discrete curve evaluates the
    fitness at the original support points; an interpolating
    curve would clamp to the last support point's target value.
    """

    support_inputs_deg = (0.0, 10.0, 20.0)
    support_outputs_deg = (10.0, 12.0, 14.0)

    inputs_rad = tuple(math.radians(a) for a in support_inputs_deg)

    csv_file = tmp_path / "target_curve.csv"
    csv_file.write_text(
        "input_angle,output_angle,weight\n0,10,1\n10,12,9\n20,14,1\n",
        encoding="utf-8",
    )

    from simulation.simulation_result import (
        SimulationResult,
    )

    first_stage = SimulationResult(
        input_angles=inputs_rad,
        output_angles=inputs_rad,
        success=True,
    )

    # Second-stage input angles are shifted far away from
    # the target support inputs (as happens with coupled
    # levers); the final outputs stay at the support inputs.
    second_stage = SimulationResult(
        input_angles=tuple(angle + math.radians(200.0) for angle in inputs_rad),
        output_angles=tuple(math.radians(a) for a in (11.0, 13.0, 15.0)),
        success=True,
    )

    simulation = (first_stage, second_stage)

    weighted_curve = TargetCurve.from_csv_strict(
        csv_file,
    )

    unweighted_curve = TargetCurve.from_points_strict(
        inputs_rad,
        tuple(math.radians(a) for a in support_outputs_deg),
    )

    fitness_weighted = CurveFitness(
        target_curve=weighted_curve,
        motion_start=inputs_rad[0],
        motion_range=inputs_rad[-1] - inputs_rad[0],
    ).evaluate(simulation)

    fitness_unweighted = CurveFitness(
        target_curve=unweighted_curve,
        motion_start=inputs_rad[0],
        motion_range=inputs_rad[-1] - inputs_rad[0],
    ).evaluate(simulation)

    # Same outputs, same targets - only the weights differ
    # (weight 9 at the middle support point).  The weighted
    # fitness must therefore differ from the unweighted one.
    assert fitness_weighted != fitness_unweighted

    # And it must be the WEIGHTED mean absolute error, not a
    # clamped/interpolated value:
    errors_deg = (1.0, 1.0, 1.0)
    expected = sum(
        weight * math.radians(err)
        for weight, err in zip(
            weighted_curve.weights,
            errors_deg,
        )
    ) / sum(weighted_curve.weights)

    assert math.isclose(
        fitness_weighted,
        expected,
        rel_tol=1e-12,
    )
