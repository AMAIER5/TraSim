"""
tests/test_fitness.py

Tests for fitness evaluation.
"""

from __future__ import annotations

import math

from analysis.fitness import (
    Fitness,
)
from analysis.target_curve import (
    TargetCurve,
)
from analysis.transfer_curve import (
    TransferCurve,
)


def create_target() -> TargetCurve:

    return TargetCurve(
        function=lambda x: x,
    )


def test_perfect_curve_has_zero_fitness():

    fitness = Fitness(
        target=create_target(),
    )

    curve = TransferCurve(
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

    result = fitness.evaluate(curve)

    assert math.isclose(
        result,
        0.0,
    )


def test_wrong_curve_has_higher_fitness():

    fitness = Fitness(
        target=create_target(),
    )

    curve = TransferCurve(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            0.0,
            2.0,
            4.0,
        ),
    )

    result = fitness.evaluate(curve)

    assert result > 0.0


def test_custom_error_metric_can_be_used():

    class DummyMetric:

        def calculate(
            self,
            curve: TransferCurve,
        ) -> float:
            return 42.0

    fitness = Fitness(
        target=create_target(),
        metric=DummyMetric(),
    )

    curve = TransferCurve(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            0.0,
            1.0,
        ),
    )

    assert fitness.evaluate(curve) == 42.0