"""tests/test_mixed_brake_anchoring.py

Tests for the anchored mixed-brake semantics:

- the braked target curve stays in the raw frame of the
  unbraked curve: the data-side output difference at the
  middle support position (e.g. 60 deg) survives blending
- the output anchor offset maps the output lever's start
  angle onto the raw reference output of the unbraked
  curve at the middle support position
- multi-stage simulations are combined: drive support
  point inputs paired with the final stage's outputs,
  and a blocking stage marks the combined result blocked
"""

from __future__ import annotations

import math

import pytest

from analysis.target_curve import TargetCurve
from optimization.mixed_brake_optimizer import (
    MixedBrakeOptimizer,
)
from simulation.simulation_result import (
    SimulationResult,
)


class _Inputs:
    """Minimal MixedBrakeInputs stand-in."""

    definition = None
    input_angles = (
        math.radians(-35.0),
        math.radians(0.0),
        math.radians(35.0),
    )
    motion_start_deg = -35.0
    motion_travel_deg = 70.0
    motion_step_deg = 35.0
    brake_min_deg = 0.0
    brake_max_deg = 90.0

    def __init__(self, unbraked, braked):
        self.unbraked = unbraked
        self.braked = braked


class _OutputLever:
    id = 2
    driver = 1
    angle_start = math.radians(80.0)


class _DriveLever:
    id = 1
    driver = None
    angle_start = 0.0


class _Definition:
    """Minimal definition stand-in exposing a
    drive and an output lever."""

    levers = (_DriveLever(), _OutputLever())

    def get_lever(self, lever_id):
        for lever in self.levers:
            if lever.id == lever_id:
                return lever
        raise KeyError(lever_id)


def _example_curves():
    unbraked = TargetCurve.from_points(
        input_angles=(
            math.radians(-35.0),
            math.radians(0.0),
            math.radians(35.0),
        ),
        output_angles=(
            math.radians(-25.0),
            math.radians(0.0),
            math.radians(46.0),
        ),
    )
    braked = TargetCurve.from_points(
        input_angles=(
            math.radians(-35.0),
            math.radians(0.0),
            math.radians(35.0),
        ),
        output_angles=(
            math.radians(19.0),
            math.radians(60.0),
            math.radians(85.0),
        ),
    )
    return unbraked, braked


def test_blended_targets_preserve_data_difference():
    from gui.mixed_brake_workflow import (
        _blended_targets,
    )

    unbraked, braked = _example_curves()
    inputs = _Inputs(unbraked, braked)
    angles = inputs.input_angles
    targets = _blended_targets(
        inputs,
        (0.0, 1.0),
        input_angles=angles,
    )
    middle = angles[1]
    assert math.degrees(
        targets[0].evaluate(middle)
    ) == pytest.approx(0.0)
    assert math.degrees(
        targets[1].evaluate(middle)
    ) == pytest.approx(60.0)


def test_reference_output_is_middle_unbraked_value():
    from gui.mixed_brake_workflow import (
        _reference_output_angle,
    )

    unbraked, braked = _example_curves()
    inputs = _Inputs(unbraked, braked)
    assert math.degrees(
        _reference_output_angle(inputs)
    ) == pytest.approx(0.0)


def test_output_offset_maps_start_angle():
    from gui.mixed_brake_workflow import (
        _output_offset,
    )

    unbraked, braked = _example_curves()
    inputs = _Inputs(unbraked, braked)
    offset_deg = math.degrees(
        _output_offset(inputs, _Definition())
    )
    assert offset_deg == pytest.approx(80.0)


def test_combined_result_pairs_drive_inputs():
    first = SimulationResult(
        input_angles=(0.1, 0.2, 0.3),
        output_angles=(1.0, 2.0, 3.0),
        success=True,
    )
    last = SimulationResult(
        input_angles=(9.1, 9.2, 9.3),
        output_angles=(4.0, 5.0, 6.0),
        success=True,
    )
    combined = MixedBrakeOptimizer._combined_result(
        (first, last)
    )
    assert combined.input_angles == (
        0.1,
        0.2,
        0.3,
    )
    assert combined.output_angles == (
        4.0,
        5.0,
        6.0,
    )
    assert combined.success


def test_combined_result_reports_blocking():
    first = SimulationResult(
        input_angles=(0.1, 0.2, 0.3),
        output_angles=(1.0, 2.0, 3.0),
        success=True,
    )
    blocked = SimulationResult(
        input_angles=(9.1, 9.2),
        output_angles=(4.0, 5.0),
        success=False,
        blocked_at=9.2,
    )
    combined = MixedBrakeOptimizer._combined_result(
        (first, blocked)
    )
    assert not combined.success
    assert combined.blocked_at == 9.2
    assert len(combined.input_angles) == len(
        combined.output_angles
    )


def test_single_stage_result_unchanged():
    only = SimulationResult(
        input_angles=(0.1, 0.2),
        output_angles=(1.0, 2.0),
        success=True,
    )
    combined = MixedBrakeOptimizer._combined_result(
        (only,)
    )
    assert combined is only
