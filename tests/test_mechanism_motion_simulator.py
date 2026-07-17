"""
tests/test_mechanism_motion_simulator.py

Tests for complete mechanism motion simulation.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage

from simulation.mechanism_motion_simulator import (
    MechanismMotionSimulator,
)

from simulation.motion_range import MotionRange


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


def create_mechanism() -> Mechanism:

    return Mechanism(
        stages=(
            create_stage(),
            create_stage(),
        )
    )


def test_mechanism_motion_runs():

    simulator = MechanismMotionSimulator(
        create_mechanism()
    )

    result = simulator.run(
        MotionRange(
            start_angle=0.0,
            max_angle=math.radians(10),
            step=math.radians(5),
        )
    )

    assert result.success

    assert len(
        result.input_angles
    ) == 3


def test_stage_results_are_saved():

    simulator = MechanismMotionSimulator(
        create_mechanism()
    )

    result = simulator.run(
        MotionRange(
            start_angle=0.0,
            max_angle=math.radians(10),
            step=math.radians(5),
        )
    )

    assert len(
        result.stage_outputs
    ) == 3

    assert len(
        result.stage_outputs[0]
    ) == 2