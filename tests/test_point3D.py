"""
tests/test_point3d.py

Unit tests for Point3D.

Issue #1: Added tests verifying that imports are at
module top (LENGTH_TOLERANCE, Vector3D) and not class
attributes, and that almost_equal uses the module-level
tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.point3d import Point3D
from core.tolerance import LENGTH_TOLERANCE
from core.vector3d import Vector3D


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructor():
    p = Point3D(1.0, 2.0, 3.0)

    assert p.x == 1.0
    assert p.y == 2.0
    assert p.z == 3.0


# ---------------------------------------------------------------------------
# Point + Vector
# ---------------------------------------------------------------------------

def test_add_vector():
    p = Point3D(1, 2, 3)
    v = Vector3D(4, 5, 6)

    result = p + v

    assert result == Point3D(5, 7, 9)


# ---------------------------------------------------------------------------
# Point - Vector
# ---------------------------------------------------------------------------

def test_subtract_vector():
    p = Point3D(5, 7, 9)
    v = Vector3D(4, 5, 6)

    result = p - v

    assert result == Point3D(1, 2, 3)


# ---------------------------------------------------------------------------
# Point - Point
# ---------------------------------------------------------------------------

def test_subtract_points():
    a = Point3D(5, 7, 9)
    b = Point3D(1, 2, 3)

    result = a - b

    assert result == Vector3D(4, 5, 6)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def test_distance():
    a = Point3D(0, 0, 0)
    b = Point3D(3, 4, 0)

    assert math.isclose(
        a.distance_to(b),
        5.0,
    )


# ---------------------------------------------------------------------------
# Midpoint
# ---------------------------------------------------------------------------

def test_midpoint():
    a = Point3D(0, 0, 0)
    b = Point3D(2, 4, 6)

    midpoint = a.midpoint(b)

    assert midpoint == Point3D(1, 2, 3)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def test_translate():
    p = Point3D(1, 2, 3)
    v = Vector3D(4, 5, 6)

    translated = p.translate(v)

    assert translated == Point3D(5, 7, 9)


# ---------------------------------------------------------------------------
# Vector to
# ---------------------------------------------------------------------------

def test_vector_to():
    a = Point3D(1, 2, 3)
    b = Point3D(5, 7, 9)

    vector = a.vector_to(b)

    assert vector == Vector3D(4, 5, 6)


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------

def test_numpy_conversion():
    p = Point3D(1, 2, 3)

    array = p.as_numpy()

    assert isinstance(array, np.ndarray)

    np.testing.assert_allclose(
        array,
        np.array([1, 2, 3]),
    )


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------

def test_iterator():
    p = Point3D(1, 2, 3)

    assert tuple(p) == (1, 2, 3)


# ---------------------------------------------------------------------------
# Invalid operations
# ---------------------------------------------------------------------------

def test_point_plus_point_is_not_supported():
    a = Point3D(1, 2, 3)
    b = Point3D(4, 5, 6)

    with pytest.raises(TypeError):
        _ = a + b


# ---------------------------------------------------------------------------
# almost_equal (Issue #1)
# ---------------------------------------------------------------------------

def test_almost_equal_identical_points():
    a = Point3D(1.0, 2.0, 3.0)
    b = Point3D(1.0, 2.0, 3.0)

    assert a.almost_equal(b)


def test_almost_equal_within_tolerance():
    a = Point3D(0.0, 0.0, 0.0)
    b = Point3D(1e-10, 0.0, 0.0)

    assert a.almost_equal(b)


def test_almost_equal_outside_tolerance():
    a = Point3D(0.0, 0.0, 0.0)
    b = Point3D(1.0, 0.0, 0.0)

    assert not a.almost_equal(b)


def test_almost_equal_custom_tolerance():
    a = Point3D(0.0, 0.0, 0.0)
    b = Point3D(0.5, 0.0, 0.0)

    assert a.almost_equal(b, tolerance=1.0)
    assert not a.almost_equal(b, tolerance=0.1)


# ---------------------------------------------------------------------------
# Issue #1: Imports at module top, not class attributes
# ---------------------------------------------------------------------------

def test_length_tolerance_not_class_attribute():
    """
    Issue #1: LENGTH_TOLERANCE was imported inside the class
    body, making it a class attribute.  It must not be a
    class attribute after the fix.
    """

    assert not hasattr(Point3D, "LENGTH_TOLERANCE")


def test_vector3d_not_class_attribute():
    """
    Issue #1: Vector3D was imported inside the class body,
    making it a class attribute.  It must not be a class
    attribute after the fix (it is already imported at
    module top).
    """

    assert not hasattr(Point3D, "Vector3D")


def test_almost_equal_default_uses_module_tolerance():
    """
    Issue #1: The default tolerance for almost_equal must
    be LENGTH_TOLERANCE from core.tolerance, not a class
    attribute.
    """

    import inspect

    sig = inspect.signature(Point3D.almost_equal)

    default = sig.parameters["tolerance"].default

    assert default == LENGTH_TOLERANCE