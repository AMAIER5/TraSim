"""
tests/test_error_metric.py

Tests for transfer curve error calculation.
"""

from __future__ import annotations

import math

from analysis.error_metric import (
    ErrorMetric,
)

from analysis.transfer_curve import (
    TransferCurve,
)


def create_target_curve():

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


def test_identical_curves_have_zero_error():

    metric = ErrorMetric(
        create_target_curve()
    )

    error = metric.calculate(
        create_target_curve()
    )

    assert math.isclose(
        error,
        0.0,
    )


def test_different_curves_have_error():

    target = create_target_curve()

    actual = TransferCurve(
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

    metric = ErrorMetric(
        target
    )

    error = metric.calculate(
        actual
    )

    assert error > 0.0


def test_error_is_symmetric_for_same_points():

    target = TransferCurve(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            0.0,
            1.0,
        ),
    )

    actual = TransferCurve(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            1.0,
            2.0,
        ),
    )

    metric = ErrorMetric(
        target
    )

    error = metric.calculate(
        actual
    )

    assert math.isclose(
        error,
        1.0,
    )