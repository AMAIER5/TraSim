"""tests/test_brake_grid.py

Tests for the evenly spaced brake grid with blended
derivative target curves.
"""

from __future__ import annotations

import math

import pytest

from analysis.brake_grid import (
    BrakeGridSettings,
    blend_target_curves,
    make_brake_grid,
)
from analysis.target_curve import TargetCurve


def test_five_positions_are_even():
    grid = BrakeGridSettings(
        brake_min_deg=180.0,
        brake_max_deg=190.0,
        position_count=5,
    )
    assert grid.fractions == (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    assert grid.brake_positions_deg == (
        180.0,
        182.5,
        185.0,
        187.5,
        190.0,
    )


def test_three_positions():
    grid = BrakeGridSettings(
        brake_min_deg=0.0,
        brake_max_deg=30.0,
        position_count=3,
    )
    assert grid.brake_positions_deg == (
        0.0,
        15.0,
        30.0,
    )


def test_two_positions_are_endpoints():
    grid = BrakeGridSettings(
        brake_min_deg=5.0,
        brake_max_deg=6.0,
        position_count=2,
    )
    assert grid.fractions == (0.0, 1.0)
    assert grid.brake_positions_deg == (5.0, 6.0)


def test_make_brake_grid_radians():
    grid = BrakeGridSettings(
        brake_min_deg=180.0,
        brake_max_deg=190.0,
        position_count=5,
    )
    radians = make_brake_grid(grid)
    assert len(radians) == 5
    for deg, rad in zip(
        grid.brake_positions_deg,
        radians,
    ):
        assert rad == pytest.approx(
            math.radians(deg)
        )


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        BrakeGridSettings(
            brake_min_deg=180.0,
            brake_max_deg=190.0,
            position_count=1,
        )
    with pytest.raises(ValueError):
        BrakeGridSettings(
            brake_min_deg=190.0,
            brake_max_deg=180.0,
            position_count=5,
        )


def _curves():
    unbraked = TargetCurve.from_points(
        input_angles=(
            math.radians(0.0),
            math.radians(10.0),
        ),
        output_angles=(
            math.radians(0.0),
            math.radians(5.0),
        ),
    )
    braked = TargetCurve.from_points(
        input_angles=(
            math.radians(0.0),
            math.radians(10.0),
        ),
        output_angles=(
            math.radians(0.0),
            math.radians(10.0),
        ),
    )
    return unbraked, braked


def test_blend_interpolates_linearly():
    unbraked, braked = _curves()
    inputs = (
        math.radians(0.0),
        math.radians(10.0),
    )
    targets = blend_target_curves(
        unbraked,
        braked,
        (0.0, 0.5, 1.0),
        input_angles=inputs,
    )
    assert len(targets) == 3
    mid = targets[1].evaluate(math.radians(10.0))
    assert mid == pytest.approx(
        math.radians(7.5)
    )


def test_blend_rejects_invalid_inputs():
    unbraked, braked = _curves()
    inputs = (math.radians(0.0),)
    with pytest.raises(ValueError):
        blend_target_curves(
            unbraked,
            braked,
            (0.0, 1.0),
            input_angles=inputs,
        )
    inputs = (
        math.radians(0.0),
        math.radians(10.0),
    )
    with pytest.raises(ValueError):
        blend_target_curves(
            unbraked,
            braked,
            (-0.5, 1.0),
            input_angles=inputs,
        )
