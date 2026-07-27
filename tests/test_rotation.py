"""
tests/test_rotation.py

Unit tests for rotation utilities.
"""

from __future__ import annotations

import math

from core.rotation import Rotation
from core.vector3d import Vector3D

# ---------------------------------------------------------------------------
# Simple axis rotation
# ---------------------------------------------------------------------------

def test_rotate_vector_around_z_axis():

    vector = Vector3D(1, 0, 0)

    result = Rotation.rotate_vector(
        vector,
        Vector3D(0, 0, 1),
        math.pi / 2,
    )

    assert result.almost_equal(
        Vector3D(0, 1, 0)
    )


# ---------------------------------------------------------------------------
# Vector alignment
# ---------------------------------------------------------------------------

def test_from_two_vectors_x_to_y():

    q = Rotation.from_two_vectors(
        Vector3D(1, 0, 0),
        Vector3D(0, 1, 0),
    )

    result = q.rotate_vector(
        Vector3D(1, 0, 0)
    )

    assert result.almost_equal(
        Vector3D(0, 1, 0)
    )


def test_from_two_vectors_parallel():

    q = Rotation.from_two_vectors(
        Vector3D(1, 0, 0),
        Vector3D(1, 0, 0),
    )

    result = q.rotate_vector(
        Vector3D(1, 0, 0)
    )

    assert result.almost_equal(
        Vector3D(1, 0, 0)
    )


def test_from_two_vectors_opposite():

    q = Rotation.from_two_vectors(
        Vector3D(1, 0, 0),
        Vector3D(-1, 0, 0),
    )

    result = q.rotate_vector(
        Vector3D(1, 0, 0)
    )

    assert result.almost_equal(
        Vector3D(-1, 0, 0)
    )


# ---------------------------------------------------------------------------
# Z axis alignment
# ---------------------------------------------------------------------------

def test_align_z_axis():

    target = Vector3D(0, 1, 0)

    q = Rotation.align_z_axis(target)

    result = q.rotate_vector(
        Vector3D(0, 0, 1)
    )

    assert result.almost_equal(
        target
    )