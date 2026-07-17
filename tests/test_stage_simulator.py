"""
tests/test_stage_simulator.py

Tests for complete stage simulation.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.lever import Lever
from mechanics.stage import Stage

from simulation.motion_range import MotionRange
from simulation.stage_simulator import StageSimulator


def create_stage() -> Stage:

    input_lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )

    output_lever = Lever(
        pivot=Point3D(100, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )

    return Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=0.0,
        output_angle=0.0,
    )


# ---------------------------------------------------------------------------
# Complete simulation
# ---------------------------------------------------------------------------

def test_stage_simulator_runs_motion():

    stage = create_stage()

    simulator = StageSimulator(stage)

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(20),
        step=math.radians(5),
    )

    result = simulator.run(motion)

    assert result.success

    assert len(result.input_angles) == 5

    assert len(result.output_angles) == 5


# ---------------------------------------------------------------------------
# Input sequence preserved
# ---------------------------------------------------------------------------

def test_stage_simulator_keeps_input_angles():

    stage = create_stage()

    simulator = StageSimulator(stage)

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    result = simulator.run(motion)

    assert result.input_angles[0] == 0.0

    assert math.isclose(
        result.input_angles[-1],
        math.radians(10),
    )