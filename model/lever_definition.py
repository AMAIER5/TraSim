from __future__ import annotations

import re
from dataclasses import dataclass

from core.vector3d import Vector3D
from core.point3d import Point3D


_PIVOT_REFERENCE_PATTERN = re.compile(
    r"^(?:lever)?(?P<lever_id>\d+)$"
)


@dataclass(frozen=True, slots=True)
class PivotReference:
    """
    Parsed ``pivot_on`` reference: the pivot of the referencing
    lever (mixer lever) sits on the endpoint of lever
    ``lever_id`` (the adjusting lever).  The brake position
    follows from the adjusting lever's angle range
    (``angle_min`` ... ``angle_max``), not from the reference.
    """

    lever_id: int


def parse_pivot_reference(value: str) -> PivotReference:
    """
    Parse a ``pivot_on`` reference of the form ``lever_id``
    (e.g. ``lever3`` or ``3``; an optional ``lever`` prefix is
    accepted).

    The angle is NOT part of the reference; the old
    ``lever_id@angle_deg`` syntax is rejected explicitly.

    Raises ``ValueError`` for anything else.
    """
    if "@" in value:
        raise ValueError(
            f"invalid pivot_on reference {value!r}: the "
            "angle suffix is no longer part of the "
            "reference; the brake position now follows "
            "from the angle range (angle_min..angle_max) "
            "of the referenced lever.  Use 'lever_id' "
            "instead, e.g. 'lever3'."
        )
    match = _PIVOT_REFERENCE_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(
            f"invalid pivot_on reference {value!r}: "
            "expected 'lever_id', e.g. 'lever3'."
        )
    return PivotReference(
        lever_id=int(match.group("lever_id")),
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

    Start angle semantics (virtual drive shaft): the target
    curve's input angle describes the rotation of a VIRTUAL
    DRIVE SHAFT of the first (drive) lever.  ``angle_start``
    (the initial start angle, before optimization) describes
    the position the lever occupies when the shaft is at the
    MEAN of all prescribed input values; the initial angular
    offset between the shaft and the start lever formed by
    the optimizer is ``angle_start - mean(input values)``,
    so the displayed initial angle of the start lever is
    that mean (e.g. 0 deg).

    When ``reference_direction`` is ``None`` the lever selects one
    automatically from the dominant axis component (see
    ``Lever._auto_reference_direction``); in that case the angle
    limits are still treated as lever angles, but their physical
    meaning depends on that automatic choice.

    A lever with ``pivot_on`` (mixer lever) has no pivot of its
    own: ``pivot_x``/``pivot_y``/``pivot_z`` and
    ``axis_x``/``axis_y``/``axis_z`` are ignored for it.  Its
    pivot sits on the endpoint of the referenced lever (adjusting
    lever) and it takes over that lever's axis.
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

    # Optional per-axis limits of the lever's pivot position
    # (the admissible room of the mounting point).  Each bound
    # is optional on its own; ``None`` means unrestricted in
    # that direction.  The pivot-shift approach uses them as
    # the limits of the displaced pair pivot (braked state),
    # and the pivot search-space parameters use them instead
    # of the default symmetric offset.
    pivot_x_min: float | None = None
    pivot_x_max: float | None = None
    pivot_y_min: float | None = None
    pivot_y_max: float | None = None
    pivot_z_min: float | None = None
    pivot_z_max: float | None = None

    # Issue #127: the fully braked position of a brake lever as
    # its OWN attribute.  Historically the braked position was
    # pinned onto the ``angle_max`` bound, so the optimizer
    # always started (and effectively stayed) at the maximum
    # allowed value and the user had no way to prescribe an
    # initial braked position separately from the bound.
    # ``None`` keeps the legacy fallback (``angle_max``).
    brake_position: float | None = None

    @property
    def resolved_brake_position(self) -> float:
        """The lever's fully braked position (radians).

        The dedicated ``brake_position`` value when the lever
        has one, otherwise the legacy fallback ``angle_max``
        (backward compatible with mechanisms written before
        the attribute existed)."""
        if self.brake_position is None:
            return self.angle_max
        return self.brake_position

    @property
    def pivot_limits(
        self,
    ) -> tuple[
        tuple[float | None, float | None],
        tuple[float | None, float | None],
        tuple[float | None, float | None],
    ]:
        """Per-axis admissible limits of the pivot position
        as ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.

        ``None`` components are unrestricted.  Raises
        ``ValueError`` when a minimum exceeds its maximum."""
        limits = (
            (self.pivot_x_min, self.pivot_x_max),
            (self.pivot_y_min, self.pivot_y_max),
            (self.pivot_z_min, self.pivot_z_max),
        )
        for minimum, maximum in limits:
            if (
                minimum is not None
                and maximum is not None
                and minimum > maximum
            ):
                raise ValueError(
                    "pivot limits of lever "
                    f"{self.id}: minimum must not be "
                    "greater than maximum"
                )
        return limits

    @property
    def has_pivot_limits(self) -> bool:
        """Whether any per-axis pivot limit is defined."""
        return any(
            bound is not None
            for pair in self.pivot_limits
            for bound in pair
        )

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
