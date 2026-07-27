"""
core/quaternion.py

Immutable quaternion implementation for 3D rotations.

A quaternion is represented as:

    q = (w, x, y, z)

where:
    w : scalar part
    (x,y,z) : vector part


References
----------
Jack B. Kuipers:
    Quaternions and Rotation Sequences.
    Princeton University Press, 1999.

Ken Shoemake:
    Animating Rotation with Quaternion Curves.
    SIGGRAPH 1985.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt

import numpy as np

from core.vector3d import Vector3D

DEFAULT_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class Quaternion:
    """
    Immutable quaternion.

    Parameters
    ----------
    w : float
        Scalar component.

    x, y, z : float
        Vector component.
    """

    w: float
    x: float
    y: float
    z: float

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Quaternion:
        """
        Return identity rotation.

        q = (1,0,0,0)
        """

        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(
        cls,
        axis: Vector3D,
        angle_rad: float,
    ) -> Quaternion:
        """
        Create quaternion from axis-angle representation.

        Formula
        -------
        q = ( cos(theta/2),
              sin(theta/2)*ax,
              sin(theta/2)*ay,
              sin(theta/2)*az )

        where the axis vector must be normalized.
        """

        axis_normalized = axis.normalized()

        half_angle = angle_rad * 0.5

        s = sin(half_angle)

        return cls(
            cos(half_angle),
            axis_normalized.x * s,
            axis_normalized.y * s,
            axis_normalized.z * s,
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    def norm_squared(self) -> float:
        """
        Squared quaternion magnitude.
        """

        return (
            self.w * self.w
            + self.x * self.x
            + self.y * self.y
            + self.z * self.z
        )

    def norm(self) -> float:
        """
        Quaternion magnitude.
        """

        return sqrt(self.norm_squared())

    def normalized(self) -> Quaternion:
        """
        Return normalized quaternion.
        """

        length = self.norm()

        if length < DEFAULT_TOLERANCE:
            raise ValueError(
                "Cannot normalize zero quaternion."
            )

        return Quaternion(
            self.w / length,
            self.x / length,
            self.y / length,
            self.z / length,
        )

    # ------------------------------------------------------------------
    # Quaternion operations
    # ------------------------------------------------------------------

    def conjugate(self) -> Quaternion:
        """
        Quaternion conjugate.

        q* = (w,-x,-y,-z)
        """

        return Quaternion(
            self.w,
            -self.x,
            -self.y,
            -self.z,
        )

    def inverse(self) -> Quaternion:
        """
        Quaternion inverse.

        Formula
        -------
        q^-1 = q* / |q|²
        """

        n2 = self.norm_squared()

        if n2 < DEFAULT_TOLERANCE:
            raise ValueError(
                "Cannot invert zero quaternion."
            )

        conjugate = self.conjugate()

        return Quaternion(
            conjugate.w / n2,
            conjugate.x / n2,
            conjugate.y / n2,
            conjugate.z / n2,
        )

    def __mul__(
        self,
        other: Quaternion,
    ) -> Quaternion:
        """
        Hamilton product.

        Formula
        -------
        q = q1 * q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2

        x = w1*x2 + x1*w2 + y1*z2 - z1*y2

        y = w1*y2 - x1*z2 + y1*w2 + z1*x2

        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        """

        if not isinstance(other, Quaternion):
            return NotImplemented

        w = (
            self.w * other.w
            - self.x * other.x
            - self.y * other.y
            - self.z * other.z
        )

        x = (
            self.w * other.x
            + self.x * other.w
            + self.y * other.z
            - self.z * other.y
        )

        y = (
            self.w * other.y
            - self.x * other.z
            + self.y * other.w
            + self.z * other.x
        )

        z = (
            self.w * other.z
            + self.x * other.y
            - self.y * other.x
            + self.z * other.w
        )

        return Quaternion(w, x, y, z)

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate_vector(
        self,
        vector: Vector3D,
    ) -> Vector3D:
        """
        Rotate vector using quaternion.

        Formula
        -------
        v' = q * v * q^-1

        Vector is represented internally as:

        v = (0,x,y,z)
        """

        q_vector = Quaternion(
            0.0,
            vector.x,
            vector.y,
            vector.z,
        )

        rotated = (
            self
            * q_vector
            * self.inverse()
        )

        return Vector3D(
            rotated.x,
            rotated.y,
            rotated.z,
        )

    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.

        Formula
        -------
        R =
        |1-2(y²+z²)   2(xy-z w)   2(xz+y w)|
        |2(xy+z w)   1-2(x²+z²)  2(yz-x w)|
        |2(xz-y w)   2(yz+x w)   1-2(x²+y²)|
        """

        q = self.normalized()

        w = q.w
        x = q.x
        y = q.y
        z = q.z

        return np.array(
            [
                [
                    1 - 2 * (y*y + z*z),
                    2 * (x*y - z*w),
                    2 * (x*z + y*w),
                ],
                [
                    2 * (x*y + z*w),
                    1 - 2 * (x*x + z*z),
                    2 * (y*z - x*w),
                ],
                [
                    2 * (x*z - y*w),
                    2 * (y*z + x*w),
                    1 - 2 * (x*x + y*y),
                ],
            ],
            dtype=float,
        )