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
- cached reference direction (perpendicular to rotation axis)
- direct Rodrigues rotation

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
    # Cached normalized axis and reference direction
    #

    _normalized_axis: Vector3D = field(
        init=False,
        repr=False,
    )

    _reference_direction: Vector3D = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate input data and cache normalized axis and reference direction.
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

        object.__setattr__(
            self,
            "_reference_direction",
            self._compute_reference_direction(),
        )

    def _compute_reference_direction(self) -> Vector3D:
        """
        Compute reference direction perpendicular to rotation axis.

        Uses the dominant axis of the rotation axis to select
        a perpendicular cardinal direction:

        - If |axis_x| >= |axis_y| and |axis_x| >= |axis_z|: use Y axis (0,1,0)
        - If |axis_y| >= |axis_x| and |axis_y| >= |axis_z|: use Z axis (0,0,1)
        - Otherwise: use X axis (1,0,0)

        This ensures the reference direction is never parallel to the rotation axis,
        preventing degenerate cases where rotation would have no effect.
        """
        axis = self._normalized_axis
        ax, ay, az = abs(axis.x), abs(axis.y), abs(axis.z)

        if ax >= ay and ax >= az:
            # X is dominant, use Y as reference (perpendicular to X)
            return Vector3D(0.0, 1.0, 0.0)
        elif ay >= ax and ay >= az:
            # Y is dominant, use Z as reference (perpendicular to Y)
            return Vector3D(0.0, 0.0, 1.0)
        else:
            # Z is dominant, use X as reference (perpendicular to Z)
            return Vector3D(1.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def direction(
        self,
        angle_rad: float,
    ) -> Vector3D:
        """
        Calculate lever direction for a given angle.

        The initial lever direction is chosen to be perpendicular to
        the rotation axis. The reference direction is selected based on
        the dominant axis of the rotation axis vector:

        - If |axis_x| is largest: reference = Y axis (0,1,0)
        - If |axis_y| is largest: reference = Z axis (0,0,1)
        - If |axis_z| is largest: reference = X axis (1,0,0)

        Uses Rodrigues rotation formula:

            v' = v * cos(theta)
                + (k x v) * sin(theta)
                + k * (k dot v) * (1 - cos(theta))

        where:
            v = reference direction (perpendicular to k)
            k = normalized rotation axis
        """
        axis = self._normalized_axis
        v = self._reference_direction

        c = cos(angle_rad)
        s = sin(angle_rad)

        # Rodrigues formula
        cross = axis.cross(v)
        dot = axis.dot(v)

        return (
            v * c
            + cross * s
            + axis * dot * (1.0 - c)
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