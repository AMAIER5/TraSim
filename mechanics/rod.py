"""
mechanics/rod.py

Ideal kinematic rod with spherical joints.

A rod connects two points in space.

The mechanical constraint is:

    distance(point_a, point_b) = length

The rod has no rotational state.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.point3d import Point3D
from core.vector3d import Vector3D


@dataclass(frozen=True, slots=True)
class Rod:
    """
    Ideal rigid rod.

    Parameters
    ----------
    point_a:
        First spherical joint.

    point_b:
        Second spherical joint.

    length:
        Distance between joints.
    """

    point_a: Point3D
    point_b: Point3D
    length: float

    @classmethod
    def from_points(
        cls,
        point_a: Point3D,
        point_b: Point3D,
    ) -> Rod:
        """
        Create rod from two endpoints.

        Formula
        -------
        length = norm(point_b - point_a)
        """

        vector = point_b - point_a

        length = vector.norm()

        if length == 0:
            raise ValueError(
                "Rod length cannot be zero."
            )

        return cls(
            point_a,
            point_b,
            length,
        )

    def direction(self) -> Vector3D:
        """
        Return normalized direction from A to B.

        Formula
        -------
        direction =
            (point_b - point_a) / length
        """

        return (
            self.point_b - self.point_a
        ).normalized()
