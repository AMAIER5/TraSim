"""
tests/test_lever.py

Unit tests for mechanical Lever component.

Updated to reflect new reference direction convention:
- Reference direction is perpendicular to rotation axis
- Selected based on dominant axis: X→Y, Y→Z, Z→X
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
    """With Z-axis rotation, reference is X, so at 0 angle lever points along X."""
    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )
    end = lever.end_position(0.0)
    assert end == Point3D(100, 0, 0)

# ---------------------------------------------------------------------------
# Rotation around Z axis
# ---------------------------------------------------------------------------

def test_end_position_90_degree_z_rotation():
    """With Z-axis rotation, reference is X, so 90 deg gives Y direction."""
    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )
    end = lever.end_position(math.pi / 2)
    assert end.almost_equal(Point3D(0, 100, 0))

# ---------------------------------------------------------------------------
# Different pivot position
# ---------------------------------------------------------------------------

def test_end_position_with_offset_pivot():
    """With Z-axis rotation, reference is X, so at 0 angle lever points along X."""
    lever = Lever(
        pivot=Point3D(50, 20, 10),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )
    end = lever.end_position(0.0)
    assert end == Point3D(150, 20, 10)

# ---------------------------------------------------------------------------
# Tilted rotation axis (X-axis)
# ---------------------------------------------------------------------------

def test_end_position_rotated_axis():
    """
    With X-axis rotation, reference is Y (perpendicular to X).
    At 90 deg, Y rotated around X by 90 deg becomes Z.
    So end position is (0, 0, 100).
    """
    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(1, 0, 0),
        length=100.0,
    )
    end = lever.end_position(math.pi / 2)
    assert end.almost_equal(Point3D(0, 0, 100))

# ---------------------------------------------------------------------------
# Direction vector
# ---------------------------------------------------------------------------

def test_end_direction():
    """With Z-axis rotation, reference is X, so 90 deg gives Y direction."""
    lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=100.0,
    )
    direction = lever.direction(math.pi / 2)
    assert direction.almost_equal(Vector3D(0, 1, 0))

# ---------------------------------------------------------------------------
# Reference direction at zero angle
# ---------------------------------------------------------------------------

def test_direction_at_zero_angle_matches_reference():
    """
    At angle 0, direction equals the reference direction,
    which is perpendicular to the rotation axis.
    Tie-breaking: when all components are equal, first condition wins (X dominant -> Y).
    """
    test_cases = [
        (Vector3D(0, 0, 1), Vector3D(1, 0, 0)),   # Z dominant -> X reference
        (Vector3D(1, 0, 0), Vector3D(0, 1, 0)),   # X dominant -> Y reference
        (Vector3D(0, 1, 0), Vector3D(0, 0, 1)),   # Y dominant -> Z reference
        (Vector3D(1, 1, 1), Vector3D(0, 1, 0)),   # Tie: ax>=ay>=az -> Y reference
        (Vector3D(2, -3, 5), Vector3D(1, 0, 0)),  # Z dominant -> X reference
    ]
    for axis, expected_dir in test_cases:
        lever = Lever(pivot=Point3D(0, 0, 0), axis=axis, length=50.0)
        d = lever.direction(0.0)
        assert d.almost_equal(expected_dir, tolerance=1e-12)

# ---------------------------------------------------------------------------
# Cross-check against Quaternion
# ---------------------------------------------------------------------------

def test_direction_nontrivial_axis_matches_quaternion():
    """
    For axis (1,1,1), reference is Y (tie-breaking: X dominant -> Y).
    Verify Rodrigues formula matches Quaternion rotation.
    """
    axis = Vector3D(1, 1, 1)
    angle = math.radians(37)
    lever = Lever(pivot=Point3D(0, 0, 0), axis=axis, length=50.0)

    lever_dir = lever.direction(angle)

    # Reference for (1,1,1) is Y
    q = Quaternion.from_axis_angle(axis, angle)
    v = Vector3D(0, 1, 0)  # Y is the reference
    quat_dir = q.rotate_vector(v)

    assert lever_dir.almost_equal(quat_dir, tolerance=1e-12)

def test_direction_nontrivial_axis_multiple_angles():
    """
    For axis (1,2,3), Z is dominant, so reference is X.
    Cross-check against Quaternion for multiple angles.
    """
    axis = Vector3D(1, 2, 3)
    lever = Lever(pivot=Point3D(0, 0, 0), axis=axis, length=50.0)

    # Reference for (1,2,3) is X (Z dominant -> else case)
    v = Vector3D(1, 0, 0)

    for deg in (-180, -90, -45, 0, 30, 60, 90, 135, 180):
        angle = math.radians(deg)
        lever_dir = lever.direction(angle)
        qi = Quaternion.from_axis_angle(axis, angle)
        quat_dir = qi.rotate_vector(v)
        assert lever_dir.almost_equal(quat_dir, tolerance=1e-12)

def test_end_position_nontrivial_axis_matches_quaternion():
    """
    For axis (1,1,1), reference is Y.
    End position must match quaternion-based calculation.
    """
    axis = Vector3D(1, 1, 1)
    angle = math.radians(53)
    length = 75.0
    pivot = Point3D(10, 20, 30)

    lever = Lever(pivot=pivot, axis=axis, length=length)
    end = lever.end_position(angle)

    # Reference for (1,1,1) is Y
    q = Quaternion.from_axis_angle(axis, angle)
    v = Vector3D(0, 1, 0)
    rotated = q.rotate_vector(v)
    expected = pivot + (rotated * length)

    assert end.almost_equal(expected, tolerance=1e-9)

# ---------------------------------------------------------------------------
# Unit vector property
# ---------------------------------------------------------------------------

def test_direction_is_unit_vector():
    """Lever.direction always returns a unit vector."""
    axis = Vector3D(1, 2, 3)
    lever = Lever(pivot=Point3D(0, 0, 0), axis=axis, length=50.0)
    for deg in (0, 45, 90, 137):
        d = lever.direction(math.radians(deg))
        assert math.isclose(d.norm(), 1.0, abs_tol=1e-12)