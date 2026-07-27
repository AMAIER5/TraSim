"""
tests/test_vector3d.py

Unit tests for Vector3D.

Author:
    Koppelgetriebe Project
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.vector3d import Vector3D

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructor():
    v = Vector3D(1.0, 2.0, 3.0)

    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

def test_addition():
    a = Vector3D(1, 2, 3)
    b = Vector3D(4, 5, 6)

    assert a + b == Vector3D(5, 7, 9)


def test_subtraction():
    a = Vector3D(5, 7, 9)
    b = Vector3D(1, 2, 3)

    assert a - b == Vector3D(4, 5, 6)


def test_negation():
    v = Vector3D(1, -2, 3)

    assert -v == Vector3D(-1, 2, -3)


def test_scalar_multiplication():
    v = Vector3D(1, 2, 3)

    assert v * 2 == Vector3D(2, 4, 6)


def test_reverse_scalar_multiplication():
    v = Vector3D(1, 2, 3)

    assert 2 * v == Vector3D(2, 4, 6)


def test_scalar_division():
    v = Vector3D(2, 4, 6)

    assert v / 2 == Vector3D(1, 2, 3)


def test_division_by_zero():
    v = Vector3D(1, 2, 3)

    with pytest.raises(ZeroDivisionError):
        _ = v / 0.0


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------

def test_norm():
    v = Vector3D(3, 4, 0)

    assert math.isclose(v.norm(), 5.0)


def test_norm_squared():
    v = Vector3D(3, 4, 0)

    assert v.norm_squared() == 25.0


def test_normalization():
    v = Vector3D(3, 4, 0)

    n = v.normalized()

    assert math.isclose(n.norm(), 1.0)


def test_normalization_zero_vector():
    with pytest.raises(ValueError):
        Vector3D(0, 0, 0).normalized()


# ---------------------------------------------------------------------------
# Dot Product
# ---------------------------------------------------------------------------

def test_dot_product():
    a = Vector3D(1, 2, 3)
    b = Vector3D(4, 5, 6)

    assert a.dot(b) == 32


def test_dot_product_orthogonal():
    a = Vector3D(1, 0, 0)
    b = Vector3D(0, 1, 0)

    assert a.dot(b) == 0


# ---------------------------------------------------------------------------
# Cross Product
# ---------------------------------------------------------------------------

def test_cross_product():
    a = Vector3D(1, 0, 0)
    b = Vector3D(0, 1, 0)

    assert a.cross(b) == Vector3D(0, 0, 1)


def test_cross_parallel_vectors():
    a = Vector3D(1, 0, 0)
    b = Vector3D(2, 0, 0)

    assert a.cross(b).is_zero()


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def test_distance():
    a = Vector3D(0, 0, 0)
    b = Vector3D(3, 4, 0)

    assert math.isclose(a.distance_to(b), 5.0)


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

def test_angle_zero():
    a = Vector3D(1, 0, 0)
    b = Vector3D(2, 0, 0)

    assert math.isclose(a.angle_to(b), 0.0)


def test_angle_90_degree():
    a = Vector3D(1, 0, 0)
    b = Vector3D(0, 1, 0)

    assert math.isclose(a.angle_to(b), math.pi / 2)


def test_angle_180_degree():
    a = Vector3D(1, 0, 0)
    b = Vector3D(-1, 0, 0)

    assert math.isclose(a.angle_to(b), math.pi)


def test_angle_zero_vector():
    with pytest.raises(ValueError):
        Vector3D(0, 0, 0).angle_to(Vector3D(1, 0, 0))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def test_is_zero():
    assert Vector3D(0, 0, 0).is_zero()


def test_not_zero():
    assert not Vector3D(1, 0, 0).is_zero()


def test_almost_equal():
    a = Vector3D(1, 2, 3)
    b = Vector3D(
        1.0 + 1e-13,
        2.0,
        3.0,
    )

    assert a.almost_equal(b)


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------

def test_numpy_conversion():
    v = Vector3D(1, 2, 3)

    array = v.as_numpy()

    assert isinstance(array, np.ndarray)

    np.testing.assert_allclose(
        array,
        np.array([1, 2, 3]),
    )


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------

def test_iterator():
    v = Vector3D(1, 2, 3)

    assert tuple(v) == (1, 2, 3)