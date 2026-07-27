"""
tests/test_quaternion.py

Unit tests for Quaternion.

Author:
    Koppelgetriebe Project
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.quaternion import Quaternion
from core.vector3d import Vector3D

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_identity():
    q = Quaternion.identity()

    assert q.w == 1.0
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0


def test_from_axis_angle():
    axis = Vector3D(0, 0, 1)

    q = Quaternion.from_axis_angle(axis, math.pi / 2)

    assert math.isclose(q.norm(), 1.0)


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------

def test_norm():
    q = Quaternion(1, 2, 3, 4)

    expected = math.sqrt(30)

    assert math.isclose(q.norm(), expected)


def test_normalized():
    q = Quaternion(2, 0, 0, 0)

    n = q.normalized()

    assert math.isclose(n.norm(), 1.0)


def test_normalize_zero():
    with pytest.raises(ValueError):
        Quaternion(0, 0, 0, 0).normalized()


# ---------------------------------------------------------------------------
# Conjugate / Inverse
# ---------------------------------------------------------------------------

def test_conjugate():
    q = Quaternion(1, 2, 3, 4)

    c = q.conjugate()

    assert c == Quaternion(1, -2, -3, -4)


def test_inverse_identity():
    q = Quaternion.identity()

    assert q.inverse() == q


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_rotate_x_to_y():
    axis = Vector3D(0, 0, 1)

    q = Quaternion.from_axis_angle(
        axis,
        math.pi / 2,
    )

    v = Vector3D(1, 0, 0)

    rotated = q.rotate_vector(v)

    assert rotated.almost_equal(
        Vector3D(0, 1, 0),
        tolerance=1e-12,
    )


def test_rotate_y_to_minus_x():
    axis = Vector3D(0, 0, 1)

    q = Quaternion.from_axis_angle(
        axis,
        math.pi / 2,
    )

    v = Vector3D(0, 1, 0)

    rotated = q.rotate_vector(v)

    assert rotated.almost_equal(
        Vector3D(-1, 0, 0),
        tolerance=1e-12,
    )


def test_rotate_180_degree():
    axis = Vector3D(1, 0, 0)

    q = Quaternion.from_axis_angle(
        axis,
        math.pi,
    )

    v = Vector3D(0, 1, 0)

    rotated = q.rotate_vector(v)

    assert rotated.almost_equal(
        Vector3D(0, -1, 0),
        tolerance=1e-12,
    )


def test_rotate_zero_degree():
    axis = Vector3D(0, 0, 1)

    q = Quaternion.from_axis_angle(axis, 0.0)

    v = Vector3D(5, 2, 8)

    assert q.rotate_vector(v).almost_equal(v)


def test_rotate_full_circle():
    axis = Vector3D(0, 0, 1)

    q = Quaternion.from_axis_angle(
        axis,
        2 * math.pi,
    )

    v = Vector3D(4, 5, 6)

    assert q.rotate_vector(v).almost_equal(v)


# ---------------------------------------------------------------------------
# Matrix Conversion
# ---------------------------------------------------------------------------

def test_rotation_matrix_shape():
    q = Quaternion.identity()

    m = q.to_rotation_matrix()

    assert isinstance(m, np.ndarray)
    assert m.shape == (3, 3)


def test_identity_matrix():
    q = Quaternion.identity()

    m = q.to_rotation_matrix()

    np.testing.assert_allclose(
        m,
        np.eye(3),
    )