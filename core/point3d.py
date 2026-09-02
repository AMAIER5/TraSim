"""
core/point3d.py

Immutable 3D point implementation.

A point represents a position in 3D space.
Unlike a vector, two points cannot be added together.

Author:
    Koppelgetriebe Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np

from core.tolerance import LENGTH_TOLERANCE
from core.vector3d import Vector3D


@dataclass(frozen=True, slots=True)
class Point3D:
    """
    Immutable point in 3D space.

    Parameters
    ----------
    x : float
        X coordinate [mm]
    y : float
        Y coordinate [mm]
    z : float
        Z coordinate [mm]
    """

    x: float
    y: float
    z: float

    # ------------------------------------------------------------------
    # Python protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __array__(self, dtype=None):
        return np.asarray((self.x, self.y, self.z), dtype=dtype)

    # ------------------------------------------------------------------
    # Point / Vector operations
    # ------------------------------------------------------------------

    def __add__(self, vector: Vector3D) -> Point3D:
        """
        Translate point by a vector.
        """
        if not isinstance(vector, Vector3D):
            return NotImplemented

        return Point3D(
            self.x + vector.x,
            self.y + vector.y,
            self.z + vector.z,
        )

    @overload
    def __sub__(
        self,
        other: Point3D,
    ) -> Vector3D:
        ...


    @overload
    def __sub__(
        self,
        other: Vector3D,
    ) -> Point3D:
        ...


    def __sub__(
        self,
        other: Point3D | Vector3D,
    ) -> Vector3D | Point3D:
        """
        Supported operations

        Point - Point  -> Vector3D

        Point - Vector -> Point3D
        """

        if isinstance(other, Point3D):
            return Vector3D(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )

        if isinstance(other, Vector3D):
            return Point3D(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )

        return NotImplemented

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def distance_to(self, other: Point3D) -> float:
        """
        Euclidean distance to another point.
        """

        return (self - other).norm()

    def midpoint(self, other: Point3D) -> Point3D:
        """
        Midpoint between two points.
        """

        return Point3D(
            (self.x + other.x) * 0.5,
            (self.y + other.y) * 0.5,
            (self.z + other.z) * 0.5,
        )

    def translate(self, vector: Vector3D) -> Point3D:
        """
        Translate point by vector.
        """

        return self + vector

    def vector_to(self, other: Point3D) -> Vector3D:
        """
        Vector pointing from this point to another point.
        """

        return other - self

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def almost_equal(
        self,
        other: Point3D,
        tolerance: float = LENGTH_TOLERANCE,
    ) -> bool:
        """
        Compare two points using Euclidean distance.
        """

        difference: Vector3D = self - other

        return difference.norm() < tolerance


    def as_numpy(self) -> np.ndarray:
        """
        Return NumPy representation.
        """

        return np.array(
            [self.x, self.y, self.z],
            dtype=float,
        )