"""mechanics/pivot_shift_builder.py

Quasi-static builder for a mechanism whose brake state is
the DISPLACEMENT of the common pivot of a lever pair within
a defined room (replacement for the modelled brake lever).

Instead of a brake lever carrying the moved pivot, the pivot
itself is a design degree of freedom: the builder receives a
planar shift ``(dx, dy)`` applied to the common pivot of the
designated lever pair.  The mechanism is built statically per
shift position:

1. Reference stage rod lengths are computed at the UNSHIFTED
   (unbraked) position from the reference lever angles.  They
   stay frozen for the braked position with IDENTICAL
   coupling rods.
2. For a braked build the common pivot (and with it every
   lever sharing the pivot coordinates) is moved by the
   shift; every stage is rebuilt with the frozen rod length
   and a newly solved reference output angle on the correct
   kinematic branch (see mechanics/reference_restore.py).

Non-mountable geometry (no solution within the output angle
bounds, angles outside limits) raises ``MechanismBuildError``;
the optimizer can treat it as a penalty.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from core.point3d import Point3D
from mechanics.lever import Lever
from mechanics.mechanism import (
    Mechanism,
    MechanismLever,
)
from mechanics.reference_restore import (
    MechanismBuildError,
    restore_output_angle,
)
from mechanics.stage import Stage
from model.mechanism_definition import (
    MechanismDefinition,
    propagate_coupled_angles,
)
from validation.stage_motion_validator import (
    StageMotionValidator,
)
from validation.stage_validation_result import (
    StageValidationResult,
)

__all__ = [
    "MechanismBuildError",
    "PivotShiftBuilder",
    "resolve_pair_ids",
]


@dataclass(frozen=True, slots=True)
class _StageSpec:
    driver_id: int
    driven_id: int
    rod_length: float
    unbraked_output_angle: float
    input_angle_min: float
    input_angle_max: float
    output_angle_min: float
    output_angle_max: float


def resolve_pair_ids(
    definition: MechanismDefinition,
    shift_lever_id: int,
) -> frozenset[int]:
    """Ids of all levers sharing the pivot of the lever
    ``shift_lever_id`` (the complete lever pair on the common
    pivot)."""
    leader = definition.get_lever(shift_lever_id)
    return frozenset(
        lever.id
        for lever in definition.levers
        if PivotShiftBuilder._same_point(
            lever.pivot,
            leader.pivot,
        )
    )


class PivotShiftBuilder:
    """Build a fixed-pivot mechanism for one shift position of
    the common pivot of a lever pair.

    Parameters
    ----------
    definition:
        Mechanism definition with FIXED levers only; a lever
        with ``pivot_on`` played the role of the brake-lever
        carrier and is rejected here.
    shift_lever_id:
        Id of one lever of the shifted pair.  Every lever
        whose pivot coordinates coincide with this lever's
        pivot (the coupled partner of the pair) is shifted
        together with it.  When ``None`` the pair is resolved
        automatically; this requires exactly one group of
        levers sharing a pivot.
    rod_lengths:
        Optionally maps a driven lever id to an explicit
        (locked) rod length; such stages keep that length
        instead of the automatically computed reference
        distance.
    """

    pivot_tolerance: float = 1.0e-9

    def __init__(
        self,
        definition: MechanismDefinition,
        *,
        shift_lever_id: int | None = None,
        validator: StageMotionValidator | None = None,
        rod_lengths: Mapping[int, float] | None = None,
    ) -> None:
        self._definition = definition
        self._validator = (
            validator
            if validator is not None
            else StageMotionValidator()
        )
        self._validation_results: list[
            StageValidationResult
        ] = []
        self._shifted_ids = self._resolve_shifted_ids(
            definition,
            shift_lever_id,
        )
        self._rod_lengths = (
            dict(rod_lengths)
            if rod_lengths is not None
            else {}
        )
        self._specs = self._build_specs(
            definition,
            self._rod_lengths,
        )

    @property
    def shifted_lever_ids(self) -> frozenset[int]:
        """Ids of all levers whose pivot is displaced by the
        shift (the complete lever pair sharing the pivot)."""
        return frozenset(self._shifted_ids)

    @staticmethod
    def _resolve_shifted_ids(
        definition: MechanismDefinition,
        shift_lever_id: int | None,
    ) -> frozenset[int]:
        on = [
            lever
            for lever in definition.levers
            if lever.pivot_on is not None
        ]
        if on:
            raise MechanismBuildError(
                "the pivot-shift approach uses fixed pivots; "
                "levers with pivot_on (brake lever carrier) "
                "are not supported: "
                + ", ".join(
                    str(lever.id)
                    for lever in on
                )
            )
        groups: dict[
            tuple[float, float, float],
            list[int],
        ] = {}
        for lever in definition.levers:
            key = (
                lever.pivot.x,
                lever.pivot.y,
                lever.pivot.z,
            )
            groups.setdefault(key, []).append(
                lever.id,
            )
        pairs = {
            key: ids
            for key, ids in groups.items()
            if len(ids) > 1
        }
        if shift_lever_id is None:
            if len(pairs) != 1:
                raise MechanismBuildError(
                    "shift_lever_id is required: found "
                    f"{len(pairs)} lever groups sharing a "
                    "pivot, so the shifted pair is ambiguous"
                )
            ids = next(iter(pairs.values()))
            shift_lever_id = ids[0]
        else:
            lever = definition.get_lever(
                shift_lever_id,
            )
            key = (
                lever.pivot.x,
                lever.pivot.y,
                lever.pivot.z,
            )
            if key not in pairs:
                raise MechanismBuildError(
                    f"lever {shift_lever_id} shares its "
                    "pivot with no other lever; the shift "
                    "requires a lever pair on a common "
                    "pivot"
                )
        return resolve_pair_ids(
            definition,
            shift_lever_id,
        )

    @classmethod
    def _same_point(
        cls,
        left: Point3D,
        right: Point3D,
    ) -> bool:
        return (
            abs(left.x - right.x)
            <= cls.pivot_tolerance
            and abs(left.y - right.y)
            <= cls.pivot_tolerance
            and abs(left.z - right.z)
            <= cls.pivot_tolerance
        )

    @staticmethod
    def _build_specs(
        definition: MechanismDefinition,
        rod_lengths: Mapping[int, float],
    ) -> tuple[_StageSpec, ...]:
        definitions = {
            lever.id: lever
            for lever in definition.levers
        }
        driven_ids = {
            lever.id
            for lever in definition.levers
            if lever.driver is not None
        }
        unknown = sorted(
            set(rod_lengths) - driven_ids
        )
        if unknown:
            raise MechanismBuildError(
                "rod_lengths reference unknown driven "
                f"levers: {unknown}"
            )
        specs = []
        for lever in definition.levers:
            if lever.driver is None:
                continue
            driver = definitions[lever.driver]
            input_endpoint = Lever(
                pivot=driver.pivot,
                axis=driver.axis,
                length=driver.length_start,
                reference_direction=(
                    driver.reference_direction
                ),
            ).end_position(driver.angle_start)
            output_endpoint = Lever(
                pivot=lever.pivot,
                axis=lever.axis,
                length=lever.length_start,
                reference_direction=(
                    lever.reference_direction
                ),
            ).end_position(lever.angle_start)
            specs.append(
                _StageSpec(
                    driver_id=driver.id,
                    driven_id=lever.id,
                    rod_length=(
                        rod_lengths[lever.id]
                        if lever.id in rod_lengths
                        else (
                            output_endpoint
                            - input_endpoint
                        ).norm()
                    ),
                    unbraked_output_angle=(
                        lever.angle_start
                    ),
                    input_angle_min=(
                        driver.angle_min
                    ),
                    input_angle_max=(
                        driver.angle_max
                    ),
                    output_angle_min=(
                        lever.angle_min
                    ),
                    output_angle_max=(
                        lever.angle_max
                    ),
                )
            )
        return tuple(specs)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        shift: tuple[float, float] = (0.0, 0.0),
    ) -> Mechanism:
        """Build the mechanism with the common pair pivot
        displaced by ``shift = (dx, dy)`` (millimetres).

        ``shift = (0, 0)`` is the unbraked position; the rod
        lengths are always those of the unbraked reference, so
        the braked mechanism differs from the unbraked one
        ONLY in the displaced pivot.
        """
        definition = self._definition
        levers: dict[int, Lever] = {}
        for lever in definition.levers:
            if lever.id in self._shifted_ids:
                pivot = replace(
                    lever.pivot,
                    x=lever.pivot.x + shift[0],
                    y=lever.pivot.y + shift[1],
                )
            else:
                pivot = lever.pivot
            levers[lever.id] = Lever(
                pivot=pivot,
                axis=lever.axis,
                length=lever.length_start,
                reference_direction=(
                    lever.reference_direction
                ),
            )

        current_angles: dict[int, float] = {
            lever.id: lever.angle_start
            for lever in definition.levers
        }

        # Issue #149: the stages are re-solved until the
        # torsion-coupled (shaft) levers are consistent with
        # their partners, so the relative angle between
        # coupled levers is the SAME in the unbraked and the
        # braked build (the braked position differs from the
        # unbraked one only in the displaced pair pivot).
        drive_id = (
            self._specs[0].driver_id
            if self._specs
            else None
        )
        stages: list[Stage] = []
        for _ in range(len(definition.levers) + 1):
            previous = dict(current_angles)
            stages = []
            for spec in self._specs:
                input_lever = levers[spec.driver_id]
                output_lever = levers[spec.driven_id]
                input_angle = current_angles[
                    spec.driver_id
                ]
                output_angle = restore_output_angle(
                    input_lever=input_lever,
                    output_lever=output_lever,
                    input_angle=input_angle,
                    rod_length=spec.rod_length,
                    reference_angle=(
                        spec.unbraked_output_angle
                    ),
                    output_angle_min=(
                        spec.output_angle_min
                    ),
                    output_angle_max=(
                        spec.output_angle_max
                    ),
                )
                current_angles[spec.driven_id] = (
                    output_angle
                )
                stages.append(
                    Stage(
                        input_lever=input_lever,
                        output_lever=output_lever,
                        rod_length=spec.rod_length,
                        input_angle_min=(
                            spec.input_angle_min
                        ),
                        input_angle_max=(
                            spec.input_angle_max
                        ),
                        output_angle_min=(
                            spec.output_angle_min
                        ),
                        output_angle_max=(
                            spec.output_angle_max
                        ),
                        input_angle=input_angle,
                        output_angle=output_angle,
                        input_endpoint=(
                            input_lever.end_position(
                                input_angle
                            )
                        ),
                        output_endpoint=(
                            output_lever.end_position(
                                output_angle
                            )
                        ),
                    )
                )
            propagate_coupled_angles(
                definition,
                current_angles,
                skip_ids=(
                    (drive_id,)
                    if drive_id is not None
                    else ()
                ),
            )
            if current_angles == previous:
                break

        self._validation_results = []
        for index, stage in enumerate(stages):
            result = self._validator.validate(
                stage,
                stage_id=index,
            )
            self._validation_results.append(result)
        positions = {
            lever.id: position
            for position, lever in enumerate(
                definition.levers,
                start=1,
            )
        }
        mechanism_levers = tuple(
            MechanismLever(
                lever=levers[lever.id],
                reference_angle=(
                    current_angles[lever.id]
                ),
                coupled_to=(
                    positions[lever.coupled]
                    if lever.coupled is not None
                    else None
                ),
            )
            for lever in definition.levers
        )
        return Mechanism(
            stages=tuple(stages),
            levers=mechanism_levers,
        )

    def rod_lengths(self) -> dict[int, float]:
        """The frozen (unbraked) rod lengths per driven lever
        id."""
        return {
            spec.driven_id: spec.rod_length
            for spec in self._specs
        }

    def get_validation_results(
        self,
    ) -> tuple[StageValidationResult, ...]:
        return tuple(self._validation_results)
