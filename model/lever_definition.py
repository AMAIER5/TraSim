from __future__ import annotations

import re
from dataclasses import dataclass

from core.vector3d import Vector3D
from core.point3d import Point3D


_PIVOT_REFERENCE_PATTERN = re.compile(
    r"^(?:lever)?(?P<lever_id>-?\d+)"
    r"@(?P<angle_deg>-?\d+(?:\.\d+)?)$"
)


@dataclass(frozen=True, slots=True)
class PivotReference:
    """
    Parsed ``pivot_on`` reference: the pivot of the referencing
    lever sits on the endpoint of lever ``lever_id`` at lever
    angle ``angle_deg`` (degrees).
    """

    lever_id: int
    angle_deg: float


def parse_pivot_reference(value: str) -> PivotReference:
    """
    Parse a ``pivot_on`` reference of the form
    ``lever_id@angle_deg`` (e.g. ``lever3@180`` or
    ``3@180``; an optional ``lever`` prefix is accepted).

    Raises ``ValueError`` for anything else.
    """
    match = _PIVOT_REFERENCE_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(
            f"invalid pivot_on reference {value!r}: "
            "expected 'lever_id@angle_deg', "
            "e.g. 'lever3@180'."
        )
    return PivotReference(
        lever_id=int(match.group("lever_id")),
        angle_deg=float(match.group("angle_deg")),
    )


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
    pivot_on: str | None = None

    @property
    def pivot_reference(self) -> PivotReference | None:
        """
        Parsed ``pivot_on`` reference, or ``None`` when the
        lever has a fixed pivot.
        """
        if self.pivot_on is None:
            return None
        return parse_pivot_reference(self.pivot_on)

    @property
    def is_driver(self) -> bool:
        return self.driver is not None

    @property
    def is_coupled(self) -> bool:
        return self.coupled is not None
