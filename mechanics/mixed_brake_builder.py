"""mechanics/mixed_brake_builder.py

Quasi-static builder for a mixer lever whose pivot sits on
the endpoint of a brake lever (``pivot_on``).

The brake lever rotates by ``beta`` and carries the pivot of
the mixer lever on its endpoint.  The mechanism is built
statically per brake position:

1. Reference stage rod lengths are computed at ``beta = 0``
   (unbraked) from the reference lever angles.  They stay
   frozen for all brake positions.
2. For ``beta != 0`` the carrier endpoint (and therefore the
   mixer pivot) moves; every affected stage is rebuilt with
   the frozen rod length and a newly solved reference output
   angle.  Restored angles propagate down the chain: a
   stage's input angle is the current (possibly restored)
   angle of its driving lever.
3. The reference output angle is restored on the correct
   kinematic branch: a fine grid scan over the admissible
   output angle range collects *all* solutions, they are
   sorted by distance to the unbraked reference angle, and
   the closest one is refined with Newton iterations down
   to the StageSimulator tolerance (< 1e-9).  A plain
   AngleSolver/predict_output approach would jump to the
   wrong branch.

Non-mountable geometry (no solution within the output angle
bounds, stretched rod, angles outside limits) raises
``MechanismBuildError``; the optimizer can treat it as a
penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from model.mechanism_definition import MechanismDefinition
from solver.constraints import rod_length_error
from validation.stage_motion_validator import (
    StageMotionValidator,
)
from validation.stage_validation_result import (
    StageValidationResult,
)


class MechanismBuildError(Exception):
    """Raised when a mechanism cannot be built for the
    requested brake position.

    Typical causes are unreachable geometry (rod too short or
    too long for the moved pivot) or reference angles outside
    the admissible lever-angle segments.
    """


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


class MixedBrakeBuilder:
    """Build a mixer mechanism for one brake position.

    The mechanism is rebuilt per brake angle ``beta`` (lever
    angle of the brake lever).  Rod lengths are frozen at
    their ``beta = 0`` reference values; only the moved mixer
    pivot and the restored reference angles change between
    builds.

    Reference rod lengths are computed per builder instance
    (never cached globally), so every optimization candidate
    gets its own frozen rods.
    """

    search_step = math.radians(0.05)
    newton_tolerance = 1.0e-13
    restore_tolerance = 1.0e-9
    newton_max_iterations = 60

    def __init__(
        self,
        definition: MechanismDefinition,
        *,
        brake_lever_id: int | None = None,
        mixer_lever_id: int | None = None,
        validator: StageMotionValidator | None = None,
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
        self._brake_id, self._mixer_id = (
            self._resolve_roles(
                definition,
                brake_lever_id,
                mixer_lever_id,
            )
        )
        self._validate_pivot_topology(definition)
        self._specs = self._build_specs(definition)

    # ------------------------------------------------------------------
    # Role resolution and topology
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_roles(
        definition: MechanismDefinition,
        brake_lever_id: int | None,
        mixer_lever_id: int | None,
    ) -> tuple[int, int]:
        mixer_candidates = [
            lever
            for lever in definition.levers
            if lever.pivot_on is not None
        ]
        if mixer_lever_id is None:
            if len(mixer_candidates) != 1:
                raise MechanismBuildError(
                    "exactly one lever with pivot_on is "
                    "required to identify the mixer lever; "
                    f"found {len(mixer_candidates)}"
                )
            mixer = mixer_candidates[0]
        else:
            mixer = definition.get_lever(mixer_lever_id)
            if mixer.pivot_on is None:
                raise MechanismBuildError(
                    f"lever {mixer.id} has no pivot_on "
                    "reference and cannot be a mixer lever"
                )
        if brake_lever_id is None:
            brake = definition.get_lever(
                mixer.pivot_reference.lever_id
            )
        else:
            brake = definition.get_lever(brake_lever_id)
            if brake.id != mixer.pivot_reference.lever_id:
                raise MechanismBuildError(
                    f"brake lever {brake.id} does not match "
                    "the mixer pivot_on reference lever "
                    f"{mixer.pivot_reference.lever_id}"
                )
        return brake.id, mixer.id

    @staticmethod
    def _validate_pivot_topology(
        definition: MechanismDefinition,
    ) -> None:
        by_id = {
            lever.id: lever
            for lever in definition.levers
        }
        for lever in definition.levers:
            if lever.pivot_on is None:
                continue
            reference = lever.pivot_reference
            parent = by_id.get(reference.lever_id)
            if parent is None:
                raise MechanismBuildError(
                    f"lever {lever.id} pivots on unknown "
                    f"lever {reference.lever_id}"
                )
            if parent.pivot_on is not None:
                raise MechanismBuildError(
                    f"lever {lever.id} pivots on lever "
                    f"{parent.id}, which has no fixed pivot; "
                    "pivot_on references must point at "
                    "levers with a fixed pivot"
                )

    # ------------------------------------------------------------------
    # Frozen reference rods (per builder instance)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_specs(
        definition: MechanismDefinition,
    ) -> tuple[_StageSpec, ...]:
        definitions = {
            lever.id: lever
            for lever in definition.levers
        }
        resolved_pivots = {}
        for lever in definition.levers:
            if lever.pivot_on is None:
                resolved_pivots[lever.id] = lever.pivot
            else:
                parent = definitions[
                    lever.pivot_reference.lever_id
                ]
                resolved_pivots[lever.id] = Lever(
                    pivot=parent.pivot,
                    axis=parent.axis,
                    length=parent.length_start,
                    reference_direction=(
                        parent.reference_direction
                    ),
                ).end_position(parent.angle_start)
        specs = []
        for lever in definition.levers:
            if lever.driver is None:
                continue
            driver = definitions[lever.driver]
            input_endpoint = Lever(
                pivot=resolved_pivots[driver.id],
                axis=driver.axis,
                length=driver.length_start,
                reference_direction=(
                    driver.reference_direction
                ),
            ).end_position(driver.angle_start)
            output_endpoint = Lever(
                pivot=resolved_pivots[lever.id],
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
                        output_endpoint - input_endpoint
                    ).norm(),
                    unbraked_output_angle=(
                        lever.angle_start
                    ),
                    input_angle_min=driver.angle_min,
                    input_angle_max=driver.angle_max,
                    output_angle_min=lever.angle_min,
                    output_angle_max=lever.angle_max,
                )
            )
        return tuple(specs)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        mixer_angle: float = 0.0,
    ) -> Mechanism:
        """Build the mechanism with the brake lever at
        ``mixer_angle`` (lever angle ``beta`` of the brake
        lever, radians)."""
        definition = self._definition
        by_id = {
            lever.id: lever
            for lever in definition.levers
        }
        brake = by_id[self._brake_id]
        if not (
            brake.angle_min
            <= mixer_angle
            <= brake.angle_max
        ):
            raise MechanismBuildError(
                f"brake angle {mixer_angle} outside "
                f"[{brake.angle_min}, {brake.angle_max}]"
            )

        current_angles: dict[int, float] = {
            lever.id: lever.angle_start
            for lever in definition.levers
        }
        current_angles[self._brake_id] = mixer_angle

        levers = self._create_levers(
            definition,
            current_angles,
        )

        stages: list[Stage] = []
        for spec in self._specs:
            input_lever = levers[spec.driver_id]
            output_lever = levers[spec.driven_id]
            input_angle = current_angles[spec.driver_id]
            if not (
                spec.input_angle_min
                <= input_angle
                <= spec.input_angle_max
            ):
                raise MechanismBuildError(
                    f"input angle {input_angle} of stage "
                    f"{spec.driver_id}->"
                    f"{spec.driven_id} outside bounds "
                    f"[{spec.input_angle_min}, "
                    f"{spec.input_angle_max}]"
                )
            output_angle = self._restore_output_angle(
                input_lever=input_lever,
                output_lever=output_lever,
                input_angle=input_angle,
                rod_length=spec.rod_length,
                reference_angle=(
                    spec.unbraked_output_angle
                ),
                output_angle_min=spec.output_angle_min,
                output_angle_max=spec.output_angle_max,
            )
            current_angles[spec.driven_id] = output_angle
            stages.append(
                Stage(
                    input_lever=input_lever,
                    output_lever=output_lever,
                    rod_length=spec.rod_length,
                    input_angle_min=spec.input_angle_min,
                    input_angle_max=spec.input_angle_max,
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

        self._validation_results = []
        for index, stage in enumerate(stages):
            result = self._validator.validate(
                stage,
                stage_id=index,
            )
            self._validation_results.append(result)
        return Mechanism(stages=tuple(stages))

    def _create_levers(
        self,
        definition: MechanismDefinition,
        current_angles: dict[int, float],
    ) -> dict[int, Lever]:
        """Create levers; pivot_on levers take their pivot
        from the current endpoint of their parent lever."""
        pending = {
            lever.id: lever
            for lever in definition.levers
        }
        levers: dict[int, Lever] = {}
        while pending:
            progressed = False
            for lever_id in list(pending):
                lever = pending[lever_id]
                if lever.pivot_on is None:
                    levers[lever_id] = Lever(
                        pivot=lever.pivot,
                        axis=lever.axis,
                        length=lever.length_start,
                        reference_direction=(
                            lever.reference_direction
                        ),
                    )
                    del pending[lever_id]
                    progressed = True
                else:
                    parent_id = (
                        lever.pivot_reference.lever_id
                    )
                    parent = levers.get(parent_id)
                    if parent is not None:
                        endpoint = parent.end_position(
                            current_angles[parent_id]
                        )
                        levers[lever_id] = Lever(
                            pivot=endpoint,
                            axis=lever.axis,
                            length=(
                                lever.length_start
                            ),
                            reference_direction=(
                                lever.reference_direction
                            ),
                        )
                        del pending[lever_id]
                        progressed = True
            if not progressed:
                raise MechanismBuildError(
                    "pivot_on references cannot be resolved"
                )
        return levers

    # ------------------------------------------------------------------
    # Branch-faithful reference restore
    # ------------------------------------------------------------------

    def _restore_output_angle(
        self,
        *,
        input_lever: Lever,
        output_lever: Lever,
        input_angle: float,
        rod_length: float,
        reference_angle: float,
        output_angle_min: float,
        output_angle_max: float,
    ) -> float:
        if not math.isfinite(output_angle_min) or not (
            math.isfinite(output_angle_max)
        ):
            output_angle_min = -math.pi
            output_angle_max = math.pi
        if output_angle_max <= output_angle_min:
            raise MechanismBuildError(
                "empty output angle range"
            )

        input_point = input_lever.end_position(input_angle)

        def residual(
            angle: float,
        ) -> float:
            return rod_length_error(
                input_point,
                output_lever.end_position(angle),
                rod_length,
            )

        if abs(residual(reference_angle)) <= (
            self.newton_tolerance
        ):
            return reference_angle

        candidates = self._collect_branches(
            residual,
            output_angle_min,
            output_angle_max,
        )
        if not candidates:
            raise MechanismBuildError(
                "no kinematic solution for the moved "
                "pivot: rod cannot be connected within "
                "the output angle bounds"
            )
        candidates.sort(
            key=lambda angle: abs(angle - reference_angle)
        )
        angle = self._newton_refine(
            residual,
            candidates[0],
            output_angle_min,
            output_angle_max,
        )
        if angle is None:
            raise MechanismBuildError(
                "reference output angle could not be "
                "restored within tolerance"
            )
        return angle

    def _collect_branches(
        self,
        residual,
        minimum: float,
        maximum: float,
    ) -> list[float]:
        step = self.search_step
        count = int(math.ceil((maximum - minimum) / step))
        candidates: list[float] = []
        previous_angle = minimum
        previous_value = residual(minimum)
        if previous_value == 0.0:
            candidates.append(minimum)
        for index in range(1, count + 1):
            angle = min(
                minimum + index * step,
                maximum,
            )
            value = residual(angle)
            if value == 0.0:
                candidates.append(angle)
            elif previous_value * value < 0.0:
                candidates.append(
                    (previous_angle + angle) / 2.0
                )
            previous_angle = angle
            previous_value = value
        return candidates

    def _newton_refine(
        self,
        residual,
        start: float,
        minimum: float,
        maximum: float,
    ) -> float | None:
        angle = start
        for _ in range(self.newton_max_iterations):
            value = residual(angle)
            if abs(value) <= self.newton_tolerance:
                return angle
            h = 1.0e-7
            slope = (residual(angle + h) - value) / h
            if slope == 0.0:
                return None
            angle = angle - value / slope
            if angle < minimum or angle > maximum:
                return None
        if abs(residual(angle)) <= self.restore_tolerance:
            return angle
        return None

    def get_validation_results(
        self,
    ) -> tuple[StageValidationResult, ...]:
        return tuple(self._validation_results)
