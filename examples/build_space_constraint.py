"""
examples/build_space_constraint.py

Demonstrates the lever-angle build-space constraint.

The lever endpoint moves on a circular arc around the pivot.
Its admissible build space is therefore described as a *circular
segment*: a reference direction (the lever's installation pose
at lever angle 0) plus a minimum and maximum lever angle, both
measured relative to that direction.  Unlike a bounding box,
this description is invariant to the orientation of the rotation
axis and requires no projection of a tilted circle into world
coordinates.

This example builds a single-stage mechanism from a CSV file
that supplies explicit ``ref_x``/``ref_y``/``ref_z`` columns, then
checks that:

  - a lever angle inside the declared segment is accepted,
  - a lever angle outside the segment is rejected by the
    stage validator (build-space violation).

User-facing angles are in DEGREES; internal calculations use
RADIANS.
"""

from __future__ import annotations

import math
from pathlib import Path

from mechanism_io.csv_reader import CsvReader

from mechanics.csv_mechanism_builder import CsvMechanismBuilder

from optimization.parameter_set import ParameterSet

from validation.stage_motion_validator import StageMotionValidator

BASE_DIR = Path(__file__).parent
MECHANISM_FILE = BASE_DIR / "mechanism_build_space.csv"


# -------------------------------------------------
# Mechanism definition (CSV with explicit reference
# directions and lever-angle segments)
# -------------------------------------------------

definition = CsvReader.read_mechanism(MECHANISM_FILE)

print("=" * 70)
print("BUILD-SPACE CONSTRAINT EXAMPLE")
print("=" * 70)

print(f"\nMechanism: {len(definition.levers)} lever(s)")
for lever in definition.levers:
    ref = lever.reference_direction
    ref_str = (
        "auto" if ref is None
        else f"({ref.x:.2f}, {ref.y:.2f}, {ref.z:.2f})"
    )
    print(
        f"  Lever {lever.id}: "
        f"angle=[{math.degrees(lever.angle_min):.1f}deg, "
        f"{math.degrees(lever.angle_max):.1f}deg] "
        f"relative to reference {ref_str}"
    )


# -------------------------------------------------
# Build the mechanism (rod length computed from the
# reference position automatically)
# -------------------------------------------------

builder = CsvMechanismBuilder(definition)
mechanism = builder.build(ParameterSet(parameters=()))
stage = mechanism.stages[0]

print("\nStage build-space segments:")
print(
    f"  Input lever:  [{math.degrees(stage.input_angle_min):.1f}deg, "
    f"{math.degrees(stage.input_angle_max):.1f}deg]"
)
print(
    f"  Output lever: [{math.degrees(stage.output_angle_min):.1f}deg, "
    f"{math.degrees(stage.output_angle_max):.1f}deg]"
)


# -------------------------------------------------
# Demonstrate the angle-segment check
# -------------------------------------------------

inside = math.radians(20)
outside = math.radians(80)

print("\nAngle-segment check (input lever):")
print(
    f"  lever angle {math.degrees(inside):.1f}deg accepted: "
    f"{stage.accepts_input_angle(inside)}"
)
print(
    f"  lever angle {math.degrees(outside):.1f}deg accepted: "
    f"{stage.accepts_input_angle(outside)}"
)


# -------------------------------------------------
# Validate the full motion range.  The validator samples
# the input lever's declared segment and reports the first
# position where the stage becomes infeasible (either
# kinematically blocked or the output lever leaves its
# declared build-space segment).
# -------------------------------------------------

validator = StageMotionValidator(steps=40)
result = validator.validate(stage, stage_id=0)

print("\nMotion-range validation:")
print(f"  valid: {result.valid}")
print(f"  checked steps: {result.checked_steps}")

if not result.valid:
    failed = result.failed_at_input_angle
    print(
        f"  first infeasible input lever angle: "
        f"{(math.degrees(failed) if failed is not None else 'n/a')}"
    )
    print(f"  reason: {result.reason}")
else:
    print("  The complete declared build space is feasible.")
