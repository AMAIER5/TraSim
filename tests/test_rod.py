"""
tests/test_rod.py

Unit tests for mechanical Rod component.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.rod import Rod


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_rod_creation_from_points():

    rod = Rod.from_points(
        Point3D(0, 0, 0),
        Point3D(100, 0, 0),
    )

    assert rod.length == 100.0


# ---------------------------------------------------------------------------
# Length calculation in 3D
# ---------------------------------------------------------------------------

def test_rod_length_calculation():

    rod = Rod.from_points(
        Point3D(0, 0, 0),
        Point3D(3, 4, 12),
    )

    assert math.isclose(
        rod.length,
        13.0,
    )


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def test_rod_direction():

    rod = Rod.from_points(
        Point3D(0, 0, 0),
        Point3D(10, 0, 0),
    )

    direction = rod.direction()

    assert direction.almost_equal(
        Vector3D(1, 0, 0)
    )


def test_rod_direction_diagonal():

    rod = Rod.from_points(
        Point3D(0, 0, 0),
        Point3D(1, 1, 1),
    )

    direction = rod.direction()

    expected = Vector3D(
        1,
        1,
        1,
    ).normalized()

    assert direction.almost_equal(
        expected
    )


# ---------------------------------------------------------------------------
# Point access
# ---------------------------------------------------------------------------

def test_rod_points_are_preserved():

    point_a = Point3D(
        10,
        20,
        30,
    )

    point_b = Point3D(
        40,
        50,
        60,
    )

    rod = Rod.from_points(
        point_a,
        point_b,
    )

    assert rod.point_a == point_a
    assert rod.point_b == point_b


# ---------------------------------------------------------------------------
# Invalid geometry
# ---------------------------------------------------------------------------

def test_zero_length_rod_is_not_allowed():

    try:
        Rod.from_points(
            Point3D(0, 0, 0),
            Point3D(0, 0, 0),
        )

    except ValueError:
        return

    assert False, "Zero length rod should raise ValueError"