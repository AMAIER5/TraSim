"""gui/manual_workflow.py

Manual (non-evolutionary) adjustment workflow for the MIXED
BRAKE (mixer) mode:

- the coupler rods (Koppelstangen) of every stage are listed
  with their automatically calculated length; each rod can be
  edited and locked in the GUI so that its length is NOT
  recalculated when a pivot (Lagerpunkt) or a lever length is
  changed manually
- the brake lever's rotation angle (Drehwinkel des
  Bremshebels) can be adjusted manually
- on demand, the achieved transmission curve (Istkurve) at
  the chosen brake angle is computed together with the
  unbraked and the fully braked target curve

The module is independent of any particular UI framework;
``gui/app.py`` (Streamlit) is a thin layer on top of it and
the unit tests exercise the workflow directly.

User I/O angles are in DEGREES, internal calculations use
RADIANS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from gui.mixed_brake_workflow import (
    MixedBrakeInputs,
    _motion,
    _output_offset,
)
from gui.workflow import WorkflowError
from mechanics.lever import Lever
from mechanics.mixed_brake_builder import (
    MechanismBuildError,
    MixedBrakeBuilder,
)
from model.lever_definition import LeverDefinition
from model.mechanism_definition import (
    MechanismDefinition,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)

__all__ = [
    "CouplerRodStage",
    "ManualCurveResult",
    "brake_lever",
    "coupler_rod_lengths",
    "coupler_rod_stages",
    "manual_transmission_curves",
]


@dataclass(frozen=True, slots=True)
class CouplerRodStage:
    """One coupler rod (Koppelstange) of the mechanism.

    The rod connects the endpoint of the driver lever
    (``driver_id``) to the endpoint of the driven lever
    (``driven_id``); ``length`` is the automatically
    calculated reference distance at ``angle_start``.
    """

    stage_index: int
    driver_id: int
    driven_id: int
    length: float


@dataclass(frozen=True, slots=True)
class ManualCurveResult:
    """Curves of one manually adjusted configuration.

    All angles are in DEGREES on the (anchored) motion grid of
    the unbraked target curve's support points, matching the
    brake-position reports.  ``target_deg`` is the blended
    target curve at ``brake_fraction`` and ``actual_deg`` the
    achieved transmission curve of the mechanism built with the
    locked rod lengths at ``brake_angle_deg``.
    """

    brake_angle_deg: float
    brake_fraction: float
    input_angles_deg: tuple[float, ...]
    unbraked_deg: tuple[float, ...]
    braked_deg: tuple[float, ...]
    target_deg: tuple[float, ...]
    actual_deg: tuple[float, ...]


def _resolved_pivots(
    definition: MechanismDefinition,
) -> dict[int, object]:
    """Pivot of every lever with pivot_on references resolved
    onto the endpoint of the referenced lever at its start
    angle (same resolution as ``MixedBrakeBuilder``)."""
    definitions = {
        lever.id: lever for lever in definition.levers
    }
    resolved: dict[int, object] = {}
    for lever in definition.levers:
        if lever.pivot_on is None:
            resolved[lever.id] = lever.pivot
        else:
            parent = definitions[
                lever.pivot_reference.lever_id
            ]
            resolved[lever.id] = Lever(
                pivot=parent.pivot,
                axis=parent.axis,
                length=parent.length_start,
                reference_direction=(
                    parent.reference_direction
                ),
            ).end_position(parent.angle_start)
    return resolved


def coupler_rod_stages(
    definition: MechanismDefinition,
) -> tuple[CouplerRodStage, ...]:
    """Coupler rods of every stage with the automatically
    calculated length at the reference pose (angle_start)."""
    definitions = {
        lever.id: lever for lever in definition.levers
    }
    resolved = _resolved_pivots(definition)
    stages: list[CouplerRodStage] = []
    for index, lever in enumerate(
        (
            lever
            for lever in definition.levers
            if lever.driver is not None
        ),
    ):
        driver = definitions[lever.driver]
        input_endpoint = Lever(
            pivot=resolved[driver.id],
            axis=driver.axis,
            length=driver.length_start,
            reference_direction=driver.reference_direction,
        ).end_position(driver.angle_start)
        output_endpoint = Lever(
            pivot=resolved[lever.id],
            axis=lever.axis,
            length=lever.length_start,
            reference_direction=lever.reference_direction,
        ).end_position(lever.angle_start)
        stages.append(
            CouplerRodStage(
                stage_index=index,
                driver_id=driver.id,
                driven_id=lever.id,
                length=(
                    output_endpoint - input_endpoint
                ).norm(),
            )
        )
    return tuple(stages)


def coupler_rod_lengths(
    definition: MechanismDefinition,
) -> dict[int, float]:
    """Mapping driven lever id -> automatically calculated
    rod length; used for every rod that is not locked."""
    return {
        stage.driven_id: stage.length
        for stage in coupler_rod_stages(definition)
    }


def brake_lever(
    definition: MechanismDefinition,
) -> LeverDefinition:
    """The adjusting (brake) lever the mixer lever pivots on.

    Raises WorkflowError when the definition has no mixer
    lever or the mixer levers disagree about the adjusting
    lever.
    """
    mixers = [
        lever
        for lever in definition.levers
        if lever.pivot_on is not None
    ]
    if len(mixers) < 1:
        raise WorkflowError(
            "Der Mischer-Modus benötigt mindestens einen "
            "Hebel mit pivot_on (Mischerhebel)."
        )
    reference_ids = {
        lever.pivot_reference.lever_id for lever in mixers
    }
    if len(reference_ids) != 1:
        raise WorkflowError(
            "Alle Hebel mit pivot_on müssen auf denselben "
            "Bremshebel verweisen; gefunden: "
            f"{sorted(reference_ids)}."
        )
    return definition.get_lever(mixers[0].pivot_reference.lever_id)


def manual_transmission_curves(
    inputs: MixedBrakeInputs,
    definition: MechanismDefinition,
    *,
    rod_lengths: Mapping[int, float],
    brake_angle_deg: float,
) -> ManualCurveResult:
    """Simulate the manually adjusted configuration.

    ``rod_lengths`` maps a driven lever id to a locked coupler
    rod length; stages without an entry recalculate their rod
    from the current geometry.  ``brake_angle_deg`` is the
    manual rotation angle of the brake lever in DEGREES.

    Returns the unbraked and fully braked target curves, the
    blended target curve at the resulting brake fraction and
    the achieved transmission curve, all on the (anchored)
    motion grid of the unbraked target curve (DEGREES).

    Raises WorkflowError with a user-facing message when the
    brake angle is inadmissible or the mechanism cannot be
    built or simulated.
    """
    brake = brake_lever(definition)
    beta = math.radians(brake_angle_deg)
    if not (
        brake.angle_min - 1.0e-12
        <= beta
        <= brake.angle_max + 1.0e-12
    ):
        raise WorkflowError(
            f"Der Drehwinkel des Bremshebels muss "
            f"zwischen {math.degrees(brake.angle_min):.1f}° "
            f"und {math.degrees(brake.angle_max):.1f}° liegen "
            f"(Hebel {brake.id})."
        )
    span = inputs.brake_max_deg - inputs.brake_min_deg
    fraction = (
        0.0
        if span <= 0.0
        else min(
            max(
                (brake_angle_deg - inputs.brake_min_deg)
                / span,
                0.0,
            ),
            1.0,
        )
    )
    try:
        builder = MixedBrakeBuilder(
            definition,
            rod_lengths=dict(rod_lengths),
        )
        mechanism = builder.build(beta)
    except MechanismBuildError as error:
        raise WorkflowError(
            "Die Mechanik kann für den Drehwinkel "
            f"{brake_angle_deg:.1f}° nicht aufgebaut werden "
            f"(Geometrie unzulässig oder blockiert): {error}"
        ) from error
    motion = _motion(inputs)
    input_angles = tuple(angle for angle in motion)
    support = (
        inputs.input_angles
        if inputs.input_angles is not None
        else input_angles
    )
    simulator = MechanismSimulator(motion=motion)
    results = simulator.simulate(mechanism)
    if not results or not all(
        result.success for result in results
    ):
        raise WorkflowError(
            "Die Mechanik blockiert im Bewegungsbereich; "
            "es gibt keine vollständige Simulation."
        )
    output_offset = _output_offset(inputs, definition)
    try:
        unbraked_deg = tuple(
            round(
                math.degrees(inputs.unbraked.evaluate(angle)),
                1,
            )
            for angle in support
        )
        braked_deg = tuple(
            round(
                math.degrees(inputs.braked.evaluate(angle)),
                1,
            )
            for angle in support
        )
    except ValueError as error:
        raise WorkflowError(
            "Die Zielkurven müssen auf denselben Stützstellen "
            f"definiert sein: {error}"
        ) from error
    target_deg = tuple(
        round(
            math.degrees(
                (1.0 - fraction)
                * inputs.unbraked.evaluate(angle)
                + fraction * inputs.braked.evaluate(angle)
            ),
            1,
        )
        for angle in support
    )
    actual_deg = tuple(
        round(
            math.degrees(angle - output_offset),
            1,
        )
        for angle in results[-1].output_angles
    )
    return ManualCurveResult(
        brake_angle_deg=round(brake_angle_deg, 1),
        brake_fraction=fraction,
        input_angles_deg=tuple(
            round(math.degrees(angle), 1)
            for angle in input_angles
        ),
        unbraked_deg=unbraked_deg,
        braked_deg=braked_deg,
        target_deg=target_deg,
        actual_deg=actual_deg,
    )
