"""
mechanics/lever_angle_range.py

Build-space (packaging) constraint for one lever.

The lever endpoint moves on a circular arc around the pivot, in
the plane perpendicular to the rotation axis.  Its admissible
build space is therefore naturally described as a *circular
segment*: a centre direction (the lever's installation pose at
angle 0) plus a minimum and maximum lever angle, both measured
relative to that direction.

Unlike a bounding box, this description is invariant to the
orientation of the rotation axis: the constructor specifies it
in the lever's own coordinate system (reference direction in
the plane of motion), and the test reduces to an angle
comparison, regardless of how the axis is oriented in world
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.vector3d import Vector3D


@dataclass(frozen=True, slots=True)
class LeverAngleRange:
    """
    Admissible lever-angle segment (circular build space).

    Parameters
    ----------
    reference_direction:
        Lever direction at lever angle 0.  Must be perpendicular
        to the lever's rotation axis.  Together with the pivot it
        defines the lever's installation pose.

    angle_min:
        Minimum admissible lever angle [rad], measured relative
        to ``reference_direction`` about the rotation axis.

    angle_max:
        Maximum admissible lever angle [rad], measured relative
        to ``reference_direction`` about the rotation axis.

    The segment spans ``[angle_min, angle_max]``.  ``angle_min``
    must not exceed ``angle_max``.
    """

    reference_direction: Vector3D

    angle_min: float
    angle_max: float

    def __post_init__(self) -> None:

        if self.angle_min > self.angle_max:
            raise ValueError(
                "angle_min must not exceed "
                "angle_max."
            )

        if self.reference_direction.norm() == 0:
            raise ValueError(
                "reference_direction must not "
                "be zero."
            )

    def contains(self, lever_angle: float) -> bool:
        """
        Return whether a lever angle lies inside the segment.
        """

        return (
            self.angle_min
            <= lever_angle
            <= self.angle_max
        )
