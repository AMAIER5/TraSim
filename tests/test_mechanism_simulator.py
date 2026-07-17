"""
tests/test_mechanism_simulator.py

Tests for multi-stage simulation.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage

from simulation.mechanism_simulator import (
    MechanismSimulator,
)


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


def test_two_stage_mechanism_creation():

    mechanism = Mechanism(
        stages=(
            create_stage(),
            create_stage(),
        )
    )

    simulator = MechanismSimulator(
        mechanism
    )

    assert simulator.mechanism == mechanism


def test_two_stage_simulation_runs():

    mechanism = Mechanism(
        stages=(
            create_stage(),
            create_stage(),
        )
    )

    simulator = MechanismSimulator(
        mechanism
    )

    result = simulator.solve(
        input_angle=0.0,
    )

    assert result.success

    assert len(result.output_angles) == 2


def test_stage_output_is_next_stage_input():

    mechanism = Mechanism(
        stages=(
            create_stage(),
            create_stage(),
        )
    )

    simulator = MechanismSimulator(
        mechanism
    )

    result = simulator.solve(
        input_angle=0.0,
    )

    assert math.isclose(
        result.stage_inputs[1],
        result.output_angles[0],
    )