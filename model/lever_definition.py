from __future__ import annotations

from dataclasses import dataclass

from core.vector3d import Vector3D
from core.point3d import Point3D


@dataclass(frozen=True, slots=True)
class LeverDefinition:
    """
    Definition of one lever read from CSV.

    The ``angle_min``/``angle_max``/``angle_start`` fields are
    lever angles: each is measured relative to the lever's
    ``reference_direction`` about its rotation ``axis``.  Together
    with the pivot they describe the lever's installation pose and
    its admissible lever-angle segment (circular build space).

    When ``reference_direction`` is ``None`` the lever selects one
    automatically from the dominant axis component (see
    ``Lever._auto_reference_direction``); in that case the angle
    limits are still treated as lever angles, but their physical
    meaning depends on that automatic choice.
    """

    id: int

    pivot: Point3D

    length_min: float
    length_max: float
    length_start: float

    angle_min: float
    angle_max: float
    angle_start: float

    axis: Vector3D

    reference_direction: Vector3D | None = None

    driver: int | None = None
    coupled: int | None = None

    @property
    def is_driver(self) -> bool:
        return self.driver is not None

    @property
    def is_coupled(self) -> bool:
        return self.coupled is not None
