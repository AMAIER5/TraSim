from __future__ import annotations

from dataclasses import dataclass

from core.vector3d import Vector3D
from core.point3d import Point3D


@dataclass(frozen=True, slots=True)
class LeverDefinition:
    """
    Definition of one lever read from CSV.
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

    driver: int | None
    coupled: int | None

    @property
    def is_driver(self) -> bool:
        return self.driver is not None

    @property
    def is_coupled(self) -> bool:
        return self.coupled is not None