"""
tests/test_constraints.py

Unit tests for geometric constraints.
"""

from __future__ import annotations

import math

from core.point3d import Point3D

from solver.constraints import (
    distance,
    rod_length_error,
)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def test_distance_zero():

    point = Point3D(1, 2, 3)

    assert distance(point, point) == 0.0


def test_distance_3d():

    point_a = Point3D(0, 0, 0)

    point_b = Point3D(3, 4, 12)

    assert math.isclose(
        distance(point_a, point_b),
        13.0,
    )


# ---------------------------------------------------------------------------
# Rod constraint
# ---------------------------------------------------------------------------

def test_rod_length_error_zero():

    point_a = Point3D(0, 0, 0)

    point_b = Point3D(100, 0, 0)

    assert math.isclose(
        rod_length_error(
            point_a,
            point_b,
            100,
        ),
        0.0,
    )


def test_rod_length_error_positive():

    point_a = Point3D(0, 0, 0)

    point_b = Point3D(120, 0, 0)

    assert math.isclose(
        rod_length_error(
            point_a,
            point_b,
            100,
        ),
        20.0,
    )


def test_rod_length_error_negative():

    point_a = Point3D(0, 0, 0)

    point_b = Point3D(80, 0, 0)

    assert math.isclose(
        rod_length_error(
            point_a,
            point_b,
            100,
        ),
        -20.0,
    )