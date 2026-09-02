"""
tests/test_lever.py

Unit tests for mechanical Lever component.

Issue #18: Added unit tests for Lever.direction with a
non-trivial rotation axis (1,1,1), cross-checked against
Quaternion.from_axis_angle + rotate_vector to verify the
Rodrigues shortcut is correct.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.quaternion import Quaternion
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


# ---------------------------------------------------------------------------
# Issue #18: Non-trivial axis cross-checked against Quaternion
# ---------------------------------------------------------------------------

def test_direction_nontrivial_axis_matches_quaternion():
    """
    Issue #18: Lever.direction uses a Rodrigues shortcut
    for v=(1,0,0).  Verify the result matches the
    general quaternion rotation
    ``Quaternion.from_axis_angle(axis, angle).rotate_vector(v)``
    for a non-trivial axis (1,1,1).
    """

    axis = Vector3D(1, 1, 1)
    angle = math.radians(37)

    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=axis,
        length=50.0,
    )

    # Lever.direction returns the rotated (1,0,0) vector.
    lever_dir = lever.direction(angle)

    # Quaternion-based reference.
    q = Quaternion.from_axis_angle(axis, angle)
    v = Vector3D(1, 0, 0)
    quat_dir = q.rotate_vector(v)

    assert lever_dir.almost_equal(
        quat_dir,
        tolerance=1e-12,
    )


def test_direction_nontrivial_axis_multiple_angles():
    """
    Issue #18: Cross-check Lever.direction against
    Quaternion rotation for several angles with axis (1,2,3).
    """

    axis = Vector3D(1, 2, 3)
    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=axis,
        length=50.0,
    )

    q = Quaternion.from_axis_angle(axis, 0)
    v = Vector3D(1, 0, 0)

    for deg in (-180, -90, -45, 0, 30, 60, 90, 135, 180):
        angle = math.radians(deg)

        lever_dir = lever.direction(angle)

        # Create a fresh quaternion for each angle.
        qi = Quaternion.from_axis_angle(axis, angle)
        quat_dir = qi.rotate_vector(v)

        assert lever_dir.almost_equal(
            quat_dir,
            tolerance=1e-12,
        )


def test_end_position_nontrivial_axis_matches_quaternion():
    """
    Issue #18: The end position (pivot + direction * length)
    must also match the quaternion-based reference for
    a non-trivial axis.
    """

    axis = Vector3D(1, 1, 1)
    angle = math.radians(53)
    length = 75.0
    pivot = Point3D(10, 20, 30)

    lever = Lever(
        pivot=pivot,
        axis=axis,
        length=length,
    )

    end = lever.end_position(angle)

    q = Quaternion.from_axis_angle(axis, angle)
    v = Vector3D(1, 0, 0)
    rotated = q.rotate_vector(v)
    expected = pivot + (rotated * length)

    assert end.almost_equal(expected, tolerance=1e-9)


def test_direction_at_zero_angle_is_unit_x():
    """
    Issue #18: At angle 0, direction must be (1,0,0)
    regardless of the axis.
    """

    for axis in (
        Vector3D(0, 0, 1),
        Vector3D(1, 0, 0),
        Vector3D(1, 1, 1),
        Vector3D(2, -3, 5),
    ):
        lever = Lever(
            pivot=Point3D(0, 0, 0),
            axis=axis,
            length=50.0,
        )

        d = lever.direction(0.0)

        assert d.almost_equal(
            Vector3D(1, 0, 0),
            tolerance=1e-12,
        )


def test_direction_is_unit_vector():
    """
    Issue #18: Lever.direction always returns a unit
    vector (since it rotates (1,0,0) by a rotation).
    """

    axis = Vector3D(1, 2, 3)
    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=axis,
        length=50.0,
    )

    for deg in (0, 45, 90, 137):
        d = lever.direction(math.radians(deg))
        assert math.isclose(d.norm(), 1.0, abs_tol=1e-12)