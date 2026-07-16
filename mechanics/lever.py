"""
mechanics/lever.py

Mechanical lever component.

A lever consists of:

- fixed pivot point
- fixed rotation axis
- lever length

The current position is calculated from the supplied angle.
The lever itself does not store a dynamic state.

This keeps kinematic solving separate from component definition.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.point3d import Point3D
from core.quaternion import Quaternion
from core.vector3d import Vector3D


@dataclass(frozen=True, slots=True)
class Lever:
    """
    Representation of a rigid lever.

    Parameters
    ----------
    pivot:
        Fixed rotation center.

    axis:
        Rotation axis.

    length:
        Lever length in mm.
    """

    pivot: Point3D
    axis: Vector3D
    length: float

    def __post_init__(self):
        """
        Validate input data.
        """

        if self.length <= 0:
            raise ValueError(
                "Lever length must be positive."
            )

        if self.axis.norm() == 0:
            raise ValueError(
                "Rotation axis cannot be zero."
            )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def direction(
        self,
        angle_rad: float,
    ) -> Vector3D:
        """
        Calculate lever direction for a given angle.

        The initial lever direction is defined along global X-axis.

        Rotation:

            direction = q * x_axis * q^-1
        """

        rotation = Quaternion.from_axis_angle(
            self.axis,
            angle_rad,
        )

        return rotation.rotate_vector(
            Vector3D(1, 0, 0)
        )

    def end_position(
        self,
        angle_rad: float,
    ) -> Point3D:
        """
        Calculate lever endpoint.

        Formula
        -------
        P_end = P_pivot + direction * length
        """

        end_vector = (
            self.direction(angle_rad)
            * self.length
        )

        return self.pivot + end_vector