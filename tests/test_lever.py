"""
tests/test_lever.py

Unit tests for mechanical Lever component.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_lever_creation():

    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )

    assert lever.length == 100.0


# ---------------------------------------------------------------------------
# Zero position
# ---------------------------------------------------------------------------

def test_end_position_zero_angle():

    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )

    end = lever.end_position(0.0)

    assert end == Point3D(
        100,
        0,
        0,
    )


# ---------------------------------------------------------------------------
# Rotation around Z axis
# ---------------------------------------------------------------------------

def test_end_position_90_degree_z_rotation():

    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )

    end = lever.end_position(
        math.pi / 2
    )

    assert end.almost_equal(
        Point3D(
            0,
            100,
            0,
        )
    )


# ---------------------------------------------------------------------------
# Different pivot position
# ---------------------------------------------------------------------------

def test_end_position_with_offset_pivot():

    lever = Lever(
        pivot=Point3D(50, 20, 10),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )

    end = lever.end_position(0.0)

    assert end == Point3D(
        150,
        20,
        10,
    )


# ---------------------------------------------------------------------------
# Tilted rotation axis
# ---------------------------------------------------------------------------

def test_end_position_rotated_axis():

    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(1, 0, 0),
        length=100.0,
    )

    end = lever.end_position(
        math.pi / 2
    )

    assert end.almost_equal(
        Point3D(
            100,
            0,
            0,
        )
    )


# ---------------------------------------------------------------------------
# Direction vector
# ---------------------------------------------------------------------------

def test_end_direction():

    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )

    direction = lever.direction(
        math.pi / 2
    )

    assert direction.almost_equal(
        Vector3D(
            0,
            1,
            0,
        )
    )