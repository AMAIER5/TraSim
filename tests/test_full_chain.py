"""
tests/test_full_chain.py

Integration test for the complete kinematic chain.
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


def create_stage(
    offset: float,
) -> Stage:
    """
    Create a simple valid stage.

    Offset allows creating independent stages.
    """

    input_lever = Lever(
        pivot=Point3D(
            offset,
            0,
            0,
        ),
        axis=Vector3D(
            0,
            0,
            1,
        ),
        length=50,
    )

    output_lever = Lever(
        pivot=Point3D(
            offset + 100,
            0,
            0,
        ),
        axis=Vector3D(
            0,
            0,
            1,
        ),
        length=50,
    )

    return Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=0.0,
        output_angle=0.0,
    )


def create_two_stage_mechanism() -> Mechanism:
    """
    Create simple two-stage mechanism.
    """

    return Mechanism(
        stages=(
            create_stage(0),
            create_stage(200),
        )
    )


def test_complete_chain_simulation():

    mechanism = create_two_stage_mechanism()

    simulator = MechanismMotionSimulator(
        mechanism
    )

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(20),
        step=math.radians(5),
    )

    result = simulator.run(
        motion
    )

    assert result.success

    assert len(
        result.input_angles
    ) == 5


def test_all_stage_outputs_are_available():

    mechanism = create_two_stage_mechanism()

    simulator = MechanismMotionSimulator(
        mechanism
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

    for step in result.stage_outputs:

        assert len(step) == 2


def test_input_angle_progression():

    mechanism = create_two_stage_mechanism()

    simulator = MechanismMotionSimulator(
        mechanism
    )

    result = simulator.run(
        MotionRange(
            start_angle=0.0,
            max_angle=math.radians(10),
            step=math.radians(5),
        )
    )

    assert math.isclose(
        result.input_angles[0],
        0.0,
    )

    assert math.isclose(
        result.input_angles[-1],
        math.radians(10),
    )