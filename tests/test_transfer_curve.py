"""
tests/test_transfer_curve.py

Tests for transfer curve analysis.
"""

from __future__ import annotations

import math

from analysis.transfer_curve import (
    TransferCurve,
)


def test_transfer_curve_creation():

    curve = TransferCurve(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            0.0,
            2.0,
        ),
    )

    assert len(
        curve.input_angles
    ) == 2

    assert len(
        curve.output_angles
    ) == 2


def test_transfer_curve_requires_equal_length():

    try:

        TransferCurve(
            input_angles=(
                0.0,
                1.0,
            ),
            output_angles=(
                0.0,
            ),
        )

        assert False

    except ValueError:

        assert True


def test_transfer_curve_interpolation():

    curve = TransferCurve(
        input_angles=(
            0.0,
            math.radians(10),
        ),
        output_angles=(
            0.0,
            math.radians(20),
        ),
    )

    result = curve.output_at(
        math.radians(5)
    )

    assert math.isclose(
        result,
        math.radians(10),
    )


def test_transfer_curve_start_and_end():

    curve = TransferCurve(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            0.0,
            2.0,
        ),
    )

    assert curve.output_at(
        0.0
    ) == 0.0

    assert curve.output_at(
        1.0
    ) == 2.0