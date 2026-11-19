"""Tests for the separate brake lever brake-position
attributes (issue #127 follow-up: "I need separate
attributes to control initial values for breaked and
unbreaked position of brake lever").

- the fully braked position is its own lever attribute
  (``brake_position``), persisted as the optional CSV column
  ``brake_angle`` (degrees); the ``angle_max`` optimization
  bound is NOT reused anymore,
- the GUI mechanism table exposes it as the editable column
  "Bremswinkel (100 %)" for the brake lever,
- the brake-effect penalty keeps the brake lever moving:
  a candidate whose braking produces the prescribed output
  change scores better than one whose travel collapses.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from analysis.target_curve import TargetCurve
from gui.mixed_brake_workflow import (
    _brake_lever_id,
    mixed_brake_inputs_from_parts,
)
from mechanism_io.csv_reader import CsvReader
from simulation.simulation_result import (
    SimulationResult,
)

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES = REPO_ROOT / "examples"

MIXER_CSV = """
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled,pivot_on
1,40,100,60,-40,40,0,0,60,0,0,0,1,,,lever3
2,20,80,50,-60,60,0,50,10,0,0,0,1,1,,
3,40,200,100,-30,30,0,180,0,0,0,0,1,2,,
4,20,80,50,-60,60,0,0,10,0,0,0,1,3,,
""".strip()

LEGACY_HEADER = (
    "id,length_min,length_max,length_start,angle_min,"
    "angle_max,angle_start,pivot_x,pivot_y,pivot_z,"
    "axis_x,axis_y,axis_z,ref_x,ref_y,ref_z,driver,"
    "coupled,pivot_on"
)


@pytest.fixture
def mixer_definition(tmp_path):
    path = tmp_path / "mixer.csv"
    path.write_text(MIXER_CSV, encoding="utf-8")
    return CsvReader.read_mechanism(path)


def _curve():
    return TargetCurve.from_points(
        input_angles=(
            math.radians(-10),
            math.radians(0),
            math.radians(10),
        ),
        output_angles=(
            math.radians(-20),
            math.radians(0),
            math.radians(20),
        ),
    )


def test_csv_round_trip_of_brake_angle(
    tmp_path,
    mixer_definition,
):
    from mechanism_io.csv_writer import CsvWriter
    from model.mechanism_definition import (
        MechanismDefinition,
    )

    brake_id = _brake_lever_id(mixer_definition)
    brake = mixer_definition.get_lever(brake_id)
    definition = MechanismDefinition(
        tuple(
            replace(
                lever,
                brake_position=(math.radians(15.0)
                                if lever.id == brake_id
                                else lever.brake_position),
            )
            for lever in mixer_definition.levers
        ),
    )
    path = tmp_path / "m.csv"
    CsvWriter.write_mechanism(definition, path)
    header = path.read_text(encoding="utf-8")
    header = header.splitlines()[0]
    assert header.startswith(LEGACY_HEADER)
    assert header.endswith("brake_angle")
    reloaded = CsvReader.read_mechanism(path)
    reloaded_brake = reloaded.get_lever(brake_id)
    assert reloaded_brake.brake_position == (
        pytest.approx(math.radians(15.0))
    )
    # bound untouched
    assert reloaded_brake.angle_max == pytest.approx(
        brake.angle_max,
    )


def test_csv_layout_unchanged_without_brake_angle(
    tmp_path,
    mixer_definition,
):
    from mechanism_io.csv_writer import CsvWriter

    path = tmp_path / "m.csv"
    CsvWriter.write_mechanism(mixer_definition, path)
    header = path.read_text(encoding="utf-8")
    assert header.splitlines()[0] == LEGACY_HEADER
    reloaded = CsvReader.read_mechanism(path)
    assert all(
        lever.brake_position is None
        for lever in reloaded.levers
    )


def test_with_mechanism_levers_reads_brake_angle(
    mixer_definition,
):
    from gui.workflow import with_mechanism_levers

    brake_id = _brake_lever_id(mixer_definition)
    brake = mixer_definition.get_lever(brake_id)
    rows = [
        {
            "id": lever.id,
            "length_min": lever.length_min,
            "length_max": lever.length_max,
            "length": lever.length_start,
            "angle_min": math.degrees(lever.angle_min),
            "angle_max": math.degrees(lever.angle_max),
            "angle": math.degrees(lever.angle_start),
            "pivot_x": lever.pivot.x,
            "pivot_y": lever.pivot.y,
            "pivot_z": lever.pivot.z,
            "axis_x": lever.axis.x,
            "axis_y": lever.axis.y,
            "axis_z": lever.axis.z,
            "driver": lever.driver,
            "coupled": lever.coupled,
            "pivot_on": lever.pivot_on,
            "brake_angle": (
                12.0
                if lever.id == brake_id
                else None
            ),
        }
        for lever in mixer_definition.levers
    ]
    base_inputs = mixed_brake_inputs_from_parts(
        mixer_definition,
        _curve(),
        _curve(),
    )
    edited = with_mechanism_levers(
        base_inputs,
        rows,
    )
    edited_brake = edited.definition.get_lever(brake_id)
    assert edited_brake.brake_position == pytest.approx(
        math.radians(12.0),
    )
    assert edited_brake.angle_max == pytest.approx(
        brake.angle_max,
    )


def test_gui_table_shows_separate_braked_position():
    """The mixed brake GUI table carries the braked position
    of the brake lever as its own column while the other
    levers show no value."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(REPO_ROOT / "gui" / "app.py"),
        default_timeout=120,
    )
    app.run()
    app.file_uploader[0].set_value(
        (
            "mechanism.csv",
            (
                EXAMPLES
                / "mixed_brake_mechanism.csv"
            ).read_bytes(),
            "text/csv",
        ),
    )
    app.file_uploader[1].set_value(
        (
            "targetcurve.csv",
            (
                EXAMPLES
                / "target_curve_unbraked.csv"
            ).read_bytes(),
            "text/csv",
        ),
    )
    app.run()
    assert not app.exception, app.exception[0].value
    braked_upload = None
    for uploader in app.file_uploader:
        if uploader.label and "voll gebremst" in str(
            uploader.label,
        ):
            braked_upload = uploader
    assert braked_upload is not None
    braked_upload.set_value(
        (
            "target_curve_braked.csv",
            (
                EXAMPLES
                / "target_curve_braked.csv"
            ).read_bytes(),
            "text/csv",
        ),
    )
    app.run()
    assert not app.exception, app.exception[0].value
    rows = app.session_state["mechanism_rows"]
    definition = _definition_from_rows(rows)
    inputs = mixed_brake_inputs_from_parts(
        definition,
        TargetCurve.from_csv_strict(
            EXAMPLES / "target_curve_unbraked.csv",
        ),
        TargetCurve.from_csv_strict(
            EXAMPLES / "target_curve_braked.csv",
        ),
    )
    brake_id = _brake_lever_id(inputs.definition)
    brake = inputs.definition.get_lever(brake_id)
    brake_row = next(
        row
        for row in rows
        if row["id"] == brake_id
    )
    assert brake_row["brake_angle"] == pytest.approx(
        math.degrees(
            brake.resolved_brake_position,
        ),
    )
    assert all(
        row["brake_angle"] is None
        for row in rows
        if row["id"] != brake_id
    )


def _definition_from_rows(rows):
    from gui.workflow import with_mechanism_levers
    from gui.mixed_brake_workflow import (
        load_mixed_brake_inputs,
    )

    base = load_mixed_brake_inputs(
        (EXAMPLES / "mixed_brake_mechanism.csv")
        .read_text(encoding="utf-8"),
        (EXAMPLES / "target_curve_unbraked.csv")
        .read_text(encoding="utf-8"),
        (EXAMPLES / "target_curve_braked.csv")
        .read_text(encoding="utf-8"),
    )
    return with_mechanism_levers(
        base,
        rows,
    ).definition


def test_brake_effect_penalty_rewards_moving_brake(
    mixer_definition,
):
    """Issue #127: "Break lever doesn't move during
    optimization."  The brake effect term must score a
    candidate whose braking really changes the output
    better than one whose brake travel collapses onto the
    unbraked position (all else equal)."""
    from analysis.brake_curve_fitness import (
        BrakeCurveFitness,
    )
    from mechanics.mixed_brake_builder import (
        MixedBrakeBuilder,
    )
    from optimization.mixed_brake_optimizer import (
        MixedBrakeOptimizer,
    )
    from simulation.mechanism_simulator import (
        MechanismSimulator,
    )
    from simulation.motion_range import MotionRange
    from simulation.simulation_result import (
        SimulationResult,
    )

    motion = MotionRange(
        start_angle=math.radians(-10),
        max_angle=math.radians(10),
        step=math.radians(2),
    )
    angles_support = (
        math.radians(-10),
        math.radians(0),
        math.radians(10),
    )
    optimizer = MixedBrakeOptimizer(
        builder=MixedBrakeBuilder(mixer_definition),
        simulator=MechanismSimulator(motion=motion),
        fitness=BrakeCurveFitness(),
        definition=mixer_definition,
        brake_positions=(
            mixer_definition.get_lever(
                _brake_lever_id(
                    mixer_definition,
                ),
            ).angle_start,
            math.radians(20.0),
        ),
        targets=(
            _curve(),
            TargetCurve.from_points(
                input_angles=angles_support,
                output_angles=(0.0, 0.1, 0.2),
            ),
        ),
        brake_effect_weight=1.0,
    )
    angles = (0.0, 0.2, 0.4)
    # target curves defined exactly on the sampled support
    # angles so their sampled outputs are exact
    unbraked_target = TargetCurve.from_points(
        input_angles=angles,
        output_angles=(-0.2, 0.0, 0.2),
    )
    braked_target = TargetCurve.from_points(
        input_angles=angles,
        output_angles=(0.0, 0.1, 0.2),
    )
    moving = optimizer._pair_brake_effect_deviation(
        [
            (
                _result(angles, (-0.2, 0.0, 0.2)),
                unbraked_target,
            ),
            (
                _result(angles, (0.0, 0.1, 0.2)),
                braked_target,
            ),
        ],
    )
    collapsed = optimizer._pair_brake_effect_deviation(
        [
            (
                _result(angles, (-0.2, 0.0, 0.2)),
                unbraked_target,
            ),
            (
                _result(angles, (-0.2, 0.0, 0.2)),
                braked_target,
            ),
        ],
    )
    # the moving brake's deviation is tiny compared to the
    # collapsed one and strictly smaller
    assert moving < collapsed
    assert moving == pytest.approx(0.0, abs=1e-2)


def _result(angles, outputs):
    return SimulationResult(
        input_angles=tuple(angles),
        output_angles=tuple(outputs),
        success=True,
        blocked_at=None,
    )
