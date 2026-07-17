"""
tests/test_blocking_behavior.py

Tests for blocking behavior during simulation.
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


def create_blocking_stage() -> Stage:
    """
    Create a stage with intentionally limited motion.

    The geometry is chosen so that the solver
    cannot find a valid solution for all angles.
    """

    input_lever = Lever(
        pivot=Point3D(
            0,
            0,
            0,
        ),
        axis=Vector3D(
            0,
            0,
            1,
        ),
        length=100,
    )

    output_lever = Lever(
        pivot=Point3D(
            120,
            0,
            0,
        ),
        axis=Vector3D(
            0,
            0,
            1,
        ),
        length=20,
    )

    return Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=0.0,
        output_angle=0.0,
    )


def test_simulation_stops_at_blocking_angle():

    mechanism = Mechanism(
        stages=(
            create_blocking_stage(),
        )
    )

    simulator = MechanismMotionSimulator(
        mechanism
    )

    result = simulator.run(
        MotionRange(
            start_angle=0.0,
            max_angle=math.radians(180),
            step=math.radians(5),
        )
    )

    assert result.success is False

    assert result.blocked_at is not None


def test_blocking_keeps_previous_results():

    mechanism = Mechanism(
        stages=(
            create_blocking_stage(),
        )
    )

    simulator = MechanismMotionSimulator(
        mechanism
    )

    result = simulator.run(
        MotionRange(
            start_angle=0.0,
            max_angle=math.radians(180),
            step=math.radians(5),
        )
    )

    assert len(
        result.input_angles
    ) > 0

    assert len(
        result.input_angles
    ) == len(
        result.stage_outputs
    )