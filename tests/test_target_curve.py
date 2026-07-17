"""
tests/test_target_curve.py

Tests for target curve generation.
"""

from __future__ import annotations

import math

import pytest

from analysis.target_curve import (
    TargetCurve,
)


def test_target_curve_from_function():

    curve = TargetCurve(
        function=lambda x: x * 2,
    )

    result = curve.evaluate(
        1.5
    )

    assert math.isclose(
        result,
        3.0,
    )


def test_target_curve_generates_transfer_curve():

    curve = TargetCurve(
        function=lambda x: x * 2,
    )

    transfer = curve.sample(
        input_angles=(
            0.0,
            1.0,
            2.0,
        )
    )

    assert transfer.input_angles == (
        0.0,
        1.0,
        2.0,
    )

    assert transfer.output_angles == (
        0.0,
        2.0,
        4.0,
    )


def test_target_curve_requires_function():

    with pytest.raises(
        TypeError
    ):

        TargetCurve(
            function=None
        )


def test_target_curve_is_immutable():

    curve = TargetCurve(
        function=lambda x: x,
    )

    with pytest.raises(
        AttributeError
    ):

        curve.function = lambda x: x * 2