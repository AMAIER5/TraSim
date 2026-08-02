"""
mechanics/lever.py

Mechanical lever component.

A lever consists of:

- fixed pivot point
- fixed rotation axis
- lever length

The current position is calculated from the supplied angle.
The lever itself does not store a dynamic state.

Performance optimization:
- cached normalized rotation axis
- direct Rodrigues rotation instead of quaternion creation

This keeps kinematic solving separate from component definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin

from core.point3d import Point3D
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

    #
    # Cached normalized axis
    #

    _normalized_axis: Vector3D = field(
        init=False,
        repr=False,
    )


    def __post_init__(self) -> None:
        """
        Validate input data and cache normalized axis.
        """

        if self.length <= 0:
            raise ValueError(
                "Lever length must be positive."
            )

        if self.axis.norm() == 0:
            raise ValueError(
                "Rotation axis cannot be zero."
            )

        object.__setattr__(
            self,
            "_normalized_axis",
            self.axis.normalized(),
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

        Uses Rodrigues rotation formula:

            v' =
                v*cos(theta)
                +
                (k x v)*sin(theta)
                +
                k*(k dot v)*(1-cos(theta))

        where:

            v = (1,0,0)

            k = normalized rotation axis
        """

        axis = self._normalized_axis

        c = cos(angle_rad)
        s = sin(angle_rad)

        #
        # Initial direction vector:
        #
        # v = (1,0,0)
        #

        cross_x = 0.0
        cross_y = axis.z
        cross_z = -axis.y

        dot = axis.x


        return Vector3D(
            c
            +
            axis.x * dot * (1.0 - c),

            cross_y * s
            +
            axis.y * dot * (1.0 - c),

            cross_z * s
            +
            axis.z * dot * (1.0 - c),
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
            *
            self.length
        )

        return self.pivot + end_vector