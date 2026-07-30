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

        curve.function = (
            lambda x: x * 2
        )


def test_target_curve_from_points():

    curve = TargetCurve.from_points(
        input_angles=(
            0.0,
            10.0,
            20.0,
        ),
        output_angles=(
            0.0,
            20.0,
            40.0,
        ),
    )

    assert math.isclose(
        curve.evaluate(5.0),
        10.0,
    )

    assert math.isclose(
        curve.evaluate(15.0),
        30.0,
    )


def test_target_curve_clamps_to_endpoints():

    curve = TargetCurve.from_points(
        input_angles=(
            0.0,
            10.0,
            20.0,
        ),
        output_angles=(
            0.0,
            20.0,
            40.0,
        ),
    )

    assert math.isclose(
        curve.evaluate(-5.0),
        0.0,
    )

    assert math.isclose(
        curve.evaluate(25.0),
        40.0,
    )


def test_target_curve_from_points_requires_same_length():

    with pytest.raises(
        ValueError
    ):

        TargetCurve.from_points(
            input_angles=(
                0.0,
                10.0,
            ),
            output_angles=(
                0.0,
            ),
        )


def test_target_curve_from_points_requires_sorted_input():

    with pytest.raises(
        ValueError
    ):

        TargetCurve.from_points(
            input_angles=(
                10.0,
                0.0,
            ),
            output_angles=(
                20.0,
                0.0,
            ),
        )


def test_target_curve_from_csv(
    tmp_path,
):

    csv_file = (
        tmp_path
        / "curve.csv"
    )

    csv_file.write_text(
        (
            "input_angle,output_angle\n"
            "0,0\n"
            "10,20\n"
            "20,40\n"
        ),
        encoding="utf-8",
    )

    curve = TargetCurve.from_csv(
        csv_file
    )

    assert math.isclose(
        curve.evaluate(5.0),
        10.0,
    )

    assert math.isclose(
        curve.evaluate(15.0),
        30.0,
    )


def test_target_curve_from_csv_requires_columns(
    tmp_path,
):

    csv_file = (
        tmp_path
        / "curve.csv"
    )

    csv_file.write_text(
        (
            "x,y\n"
            "0,0\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):

        TargetCurve.from_csv(
            csv_file
        )