"""
tests/test_curve_gradient.py

Tests for the gradient (slope) deviation term of the fitness
function (issue #65): besides the point-wise comparison of the
support points, the deviation of the curve gradient between two
support points is part of the fitness value.
"""

from __future__ import annotations

import math

from analysis.curve_fitness import (
    CurveFitness,
    gradient_deviation,
)
from analysis.target_curve import (
    TargetCurve,
)
from simulation.simulation_result import (
    SimulationResult,
)


def linear_target() -> TargetCurve:
    return TargetCurve(
        function=lambda angle: angle,
    )


def test_gradient_deviation_zero_for_identical_curve():
    deviation = gradient_deviation(
        target=linear_target(),
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
    )
    assert deviation == 0.0


def test_gradient_deviation_zero_below_two_points():
    deviation = gradient_deviation(
        target=linear_target(),
        input_angles=(1.0,),
        output_angles=(1.0,),
    )
    assert deviation == 0.0


def test_zigzag_curve_has_positive_gradient_deviation():
    deviation = gradient_deviation(
        target=linear_target(),
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 2.0, 2.0),
    )
    assert deviation > 0.0


def test_deviation_at_support_points_zigzag_is_punished():
    """A curve that hits every target point but zig-zags
    between them must receive a worse fitness than a curve
    with the same point error but a straight run."""
    fitness = CurveFitness(
        target_curve=linear_target(),
        gradient_weight=1.0,
    )

    zigzag = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(1.0, 1.0, 3.0),
        success=True,
    )
    straight = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.5, 3.0),
        success=True,
    )

    zigzag_fitness = fitness.evaluate((zigzag,))
    straight_fitness = fitness.evaluate((straight,))

    assert zigzag_fitness > straight_fitness


def test_zero_gradient_weight_restores_pointwise_fitness():
    fitness = CurveFitness(
        target_curve=linear_target(),
        gradient_weight=0.0,
    )

    result = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(1.0, 1.0, 3.0),
        success=True,
    )

    pointwise = (
        abs(1.0 - 0.0)
        + abs(1.0 - 1.0)
        + abs(3.0 - 2.0)
    ) / 3.0

    assert math.isclose(
        fitness.evaluate((result,)),
        pointwise,
        rel_tol=1e-12,
    )


def test_identical_curve_has_zero_fitness_with_gradient():
    fitness = CurveFitness(
        target_curve=linear_target(),
        gradient_weight=1.0,
    )

    result = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
        success=True,
    )

    assert fitness.evaluate((result,)) == 0.0
