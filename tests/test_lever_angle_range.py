"""
tests/test_lever_angle_range.py

Tests for the circular build-space constraint
``LeverAngleRange``.
"""

from __future__ import annotations

import math

import pytest

from core.vector3d import Vector3D
from mechanics.lever_angle_range import LeverAngleRange


def test_contains_inside_segment():
    rng = LeverAngleRange(
        reference_direction=Vector3D(1, 0, 0),
        angle_min=math.radians(-30),
        angle_max=math.radians(30),
    )

    assert rng.contains(math.radians(0))
    assert rng.contains(math.radians(10))
    assert rng.contains(math.radians(-30))
    assert rng.contains(math.radians(30))


def test_contains_outside_segment():
    rng = LeverAngleRange(
        reference_direction=Vector3D(1, 0, 0),
        angle_min=math.radians(-30),
        angle_max=math.radians(30),
    )

    assert not rng.contains(math.radians(31))
    assert not rng.contains(math.radians(-31))
    assert not rng.contains(math.radians(180))


def test_min_greater_than_max_raises():
    with pytest.raises(ValueError, match="angle_min"):
        LeverAngleRange(
            reference_direction=Vector3D(1, 0, 0),
            angle_min=math.radians(30),
            angle_max=math.radians(-30),
        )


def test_zero_reference_direction_raises():
    with pytest.raises(ValueError, match="reference_direction"):
        LeverAngleRange(
            reference_direction=Vector3D(0, 0, 0),
            angle_min=math.radians(-30),
            angle_max=math.radians(30),
        )


def test_boundary_values_accepted():
    rng = LeverAngleRange(
        reference_direction=Vector3D(0, 1, 0),
        angle_min=math.radians(-10),
        angle_max=math.radians(10),
    )

    assert rng.contains(rng.angle_min)
    assert rng.contains(rng.angle_max)
