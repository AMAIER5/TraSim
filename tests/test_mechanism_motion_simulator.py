"""
tests/test_mechanism_motion_simulator.py

Tests for complete mechanism motion simulation.

Issue #12: Count expectations are now documented with
explicit comments and cross-checked against
MotionRange.count.  The motion range semantics are:

    max_angle=radians(10), step=radians(5)
    → angles: 0°, 5°, 10° = 3 points
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
    """
    Two-stage mechanism with 3 input angles (0° to 10°
    in 5° steps) produces 3 output points.
    """

    simulator = MechanismMotionSimulator(
        create_mechanism()
    )

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    # 0°, 5°, 10° → 3 angles
    assert motion.count == 3

    result = simulator.run(
        motion
    )

    assert result.success

    assert len(
        result.input_angles
    ) == 3


def test_stage_results_are_saved():
    """
    Two-stage mechanism with 3 input angles produces
    3 stage output tuples, each containing 2 stage
    outputs.
    """

    simulator = MechanismMotionSimulator(
        create_mechanism()
    )

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    # 0°, 5°, 10° → 3 angles
    assert motion.count == 3

    result = simulator.run(
        motion
    )

    assert len(
        result.stage_outputs
    ) == 3

    assert len(
        result.stage_outputs[0]
    ) == 2


def test_single_point_motion():
    """
    Issue #12: max_angle=0 yields exactly one point
    with 2 stage outputs.
    """

    simulator = MechanismMotionSimulator(
        create_mechanism()
    )

    motion = MotionRange(
        start_angle=0.0,
        max_angle=0.0,
        step=math.radians(5),
    )

    assert motion.count == 1

    result = simulator.run(
        motion
    )

    assert result.success

    assert len(result.input_angles) == 1

    assert len(result.stage_outputs) == 1

    assert len(result.stage_outputs[0]) == 2


def test_motion_count_matches_input_angles():
    """
    Issue #12: The number of input angles in the result
    must match MotionRange.count for various step sizes.
    """

    simulator = MechanismMotionSimulator(
        create_mechanism()
    )

    for max_deg, step_deg in [
        (10, 5),   # 3 angles
        (20, 5),   # 5 angles
        (9, 3),    # 4 angles
        (0, 1),    # 1 angle
    ]:

        motion = MotionRange(
            start_angle=0.0,
            max_angle=math.radians(max_deg),
            step=math.radians(step_deg),
        )

        result = simulator.run(
            motion
        )

        assert result.success

        assert len(result.input_angles) == motion.count