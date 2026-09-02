"""
core/vector3d.py

Immutable 3D vector implementation.

This module provides the fundamental vector class used throughout the
kinematic solver.

Author:
    Koppelgetriebe Project

License:
    MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import acos, isclose, sqrt

import numpy as np

from core.tolerance import VECTOR_TOLERANCE

DEFAULT_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class Vector3D:
    """
    Immutable three-dimensional vector.

    Parameters
    ----------
    x : float
        X component.
    y : float
        Y component.
    z : float
        Z component.
    """

    x: float
    y: float
    z: float

    # ------------------------------------------------------------------
    # Python protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __array__(self, dtype=None):
        return np.asarray((self.x, self.y, self.z), dtype=dtype)

    # ------------------------------------------------------------------
    # Basic arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> Vector3D:
        return Vector3D(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )

    def __rmul__(self, scalar: float) -> Vector3D:
        return self * scalar

    def __truediv__(self, scalar: float) -> Vector3D:
        """
        Divide by a scalar.

        Issue #2: Only exact zero raises ZeroDivisionError.
        A tiny-but-nonzero scalar is a valid divisor.
        """

        if scalar == 0.0:
            raise ZeroDivisionError("Division by zero.")

        return Vector3D(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
        )

    def __abs__(self) -> float:
        return self.norm()

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def norm_squared(self) -> float:
        """
        Squared Euclidean norm.
        """
        return self.dot(self)

    def norm(self) -> float:
        """
        Return Euclidean vector length.

        Formula
        -------
        norm(v) = sqrt(x*x + y*y + z*z)
        """

        return sqrt(
            self.x * self.x
            + self.y * self.y
            + self.z * self.z
        )

    def normalized(self) -> Vector3D:
        """
        Return normalized vector.

        Raises
        ------
        ValueError
            If vector length is zero.
        """

        length = self.norm()

        if length < DEFAULT_TOLERANCE:
            raise ValueError("Cannot normalize zero vector.")

        return self / length

    def dot(self, other: Vector3D) -> float:
        """
        Dot product.
        """
        return (
            self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )

    def cross(self, other: Vector3D) -> Vector3D:
        """
        Cross product.
        """
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3D) -> float:
        """
        Euclidean distance.
        """
        return (self - other).norm()

    def angle_to(self, other: Vector3D) -> float:
        """
        Angle between vectors [rad].

        Returns
        -------
        float
            Angle in radians.

        Raises
        ------
        ValueError
            If one vector has zero length.
        """

        a = self.normalized()
        b = other.normalized()

        cosine = np.clip(a.dot(b), -1.0, 1.0)

        return float(acos(cosine))


    def __matmul__(self, other: Vector3D) -> float:
        """
        Dot product using the @ operator.
        """
        return self.dot(other)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_zero(
        self,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """
        Check whether vector is approximately zero.
        """

        return self.norm_squared() < tolerance * tolerance

    def almost_equal(
        self,
        other: Vector3D,
        tolerance: float = VECTOR_TOLERANCE,
    ) -> bool:
        """
        Geometric comparison using Euclidean distance.
        """
        return (self - other).norm() < tolerance

    def as_numpy(self) -> np.ndarray:
        """
        Return NumPy representation.
        """

        return np.array(
            [self.x, self.y, self.z],
            dtype=float,
        )