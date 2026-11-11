"""tests/test_manual_workflow.py

Tests for the manual adjustment workflow of the mixed brake
(mixer) mode used by the GUI:

- the coupler rod stages/lengths are derived from the
  mechanism definition and match the builder's reference rods
- a locked rod length is kept when a pivot changes, while an
  unlocked rod is recalculated from the new geometry
- the manual transmission curve at the brake lever's start
  angle reproduces the unbraked state, at the maximum angle
  the fully braked state
- moving a pivot with locked rods changes the achieved
  transmission curve
- an inadmissible brake angle raises a user-facing error
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from gui.manual_workflow import (
    brake_lever,
    coupler_rod_lengths,
    coupler_rod_stages,
    manual_transmission_curves,
)
from gui.mixed_brake_workflow import (
    build_mixed_brake_state,
    load_mixed_brake_inputs,
    mixed_brake_inputs_from_parts,
    mixed_brake_parameter_template,
)
from gui.workflow import WorkflowError
from mechanics.mixed_brake_builder import (
    MechanismBuildError,
    MixedBrakeBuilder,
)
from model.mechanism_definition import MechanismDefinition

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)
MECHANISM_FILE = (
    EXAMPLES_DIR / "mixed_brake_mechanism.csv"
)
UNBRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_unbraked.csv"
)
BRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_braked.csv"
)


@pytest.fixture
def inputs():
    return load_mixed_brake_inputs(
        MECHANISM_FILE.read_text(encoding="utf-8"),
        UNBRAKED_FILE.read_text(encoding="utf-8"),
        BRAKED_FILE.read_text(encoding="utf-8"),
    )


def _moved_pivot_definition(definition, dx: float = 20.0):
    """Copy of ``definition`` with lever 2's pivot moved."""
    levers = tuple(
        replace(
            lever,
            pivot=replace(
                lever.pivot,
                x=lever.pivot.x + dx,
            ),
        )
        if lever.id == 2
        else lever
        for lever in definition.levers
    )
    return MechanismDefinition(levers)


def test_brake_lever_is_the_pivot_on_reference(inputs):
    assert brake_lever(inputs.definition).id == 3


def test_coupler_rod_stages_match_builder_reference(inputs):
    stages = coupler_rod_stages(inputs.definition)
    assert len(stages) == 1
    assert stages[0].driver_id == 1
    assert stages[0].driven_id == 2
    assert stages[0].length > 0.0
    builder = MixedBrakeBuilder(inputs.definition)
    mechanism = builder.build(
        inputs.definition.get_lever(3).angle_start
    )
    assert mechanism.stages[0].rod_length == (
        pytest.approx(stages[0].length)
    )
    assert coupler_rod_lengths(inputs.definition) == {2: stages[0].length}


def test_locked_rod_length_is_kept_when_pivot_changes(inputs):
    definition = inputs.definition
    moved = _moved_pivot_definition(definition)
    original_length = coupler_rod_lengths(definition)[2]
    assert coupler_rod_lengths(moved)[2] != pytest.approx(
        original_length
    )
    locked = MixedBrakeBuilder(
        moved,
        rod_lengths={2: original_length},
    ).build(definition.get_lever(3).angle_start)
    assert locked.stages[0].rod_length == pytest.approx(
        original_length
    )
    unlocked = MixedBrakeBuilder(moved).build(
        definition.get_lever(3).angle_start
    )
    assert unlocked.stages[0].rod_length == pytest.approx(
        coupler_rod_lengths(moved)[2]
    )


def test_unknown_rod_length_lever_raises(inputs):
    with pytest.raises(MechanismBuildError):
        MixedBrakeBuilder(
            inputs.definition,
            rod_lengths={99: 10.0},
        )


def test_manual_curve_at_start_angle_matches_unbraked(inputs):
    brake = brake_lever(inputs.definition)
    start_deg = math.degrees(brake.angle_start)
    curve = manual_transmission_curves(
        inputs,
        inputs.definition,
        rod_lengths={},
        brake_angle_deg=start_deg,
    )
    assert curve.brake_fraction == pytest.approx(0.0)
    assert curve.target_deg == pytest.approx(
        curve.unbraked_deg
    )
    reference = build_mixed_brake_state(
        inputs,
        mixed_brake_parameter_template(inputs),
        brake_fraction=0.0,
    )
    assert curve.actual_deg == pytest.approx(
        reference.actual_deg
    )
    assert curve.input_angles_deg == pytest.approx(
        reference.input_angles_deg
    )


def test_manual_curve_at_max_angle_matches_braked(inputs):
    brake = brake_lever(inputs.definition)
    max_deg = math.degrees(brake.angle_max)
    curve = manual_transmission_curves(
        inputs,
        inputs.definition,
        rod_lengths={},
        brake_angle_deg=max_deg,
    )
    assert curve.brake_fraction == pytest.approx(1.0)
    assert curve.target_deg == pytest.approx(curve.braked_deg)
    reference = build_mixed_brake_state(
        inputs,
        mixed_brake_parameter_template(inputs),
        brake_fraction=1.0,
    )
    assert curve.actual_deg == pytest.approx(
        reference.actual_deg
    )


def test_moving_pivot_with_locked_rods_changes_curve(inputs):
    moved = _moved_pivot_definition(inputs.definition)
    moved_inputs = mixed_brake_inputs_from_parts(
        moved,
        inputs.unbraked,
        inputs.braked,
    )
    locked_lengths = coupler_rod_lengths(inputs.definition)
    start_deg = inputs.brake_min_deg
    locked_curve = manual_transmission_curves(
        moved_inputs,
        moved,
        rod_lengths=locked_lengths,
        brake_angle_deg=start_deg,
    )
    unlocked_curve = manual_transmission_curves(
        moved_inputs,
        moved,
        rod_lengths={},
        brake_angle_deg=start_deg,
    )
    assert locked_curve.actual_deg != pytest.approx(
        unlocked_curve.actual_deg
    )


def test_inadmissible_brake_angle_raises(inputs):
    brake = brake_lever(inputs.definition)
    with pytest.raises(WorkflowError):
        manual_transmission_curves(
            inputs,
            inputs.definition,
            rod_lengths={},
            brake_angle_deg=math.degrees(brake.angle_max) + 5.0,
        )
