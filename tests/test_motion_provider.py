"""
tests/test_motion_provider.py

Tests for MotionProvider implementations.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from simulation.adaptive_motion_range import AdaptiveMotionRange
from simulation.motion_provider import MotionProvider
from simulation.motion_range import MotionRange
from simulation.stage_simulator import StageSimulator
from solver.solver_precision import SolverPrecision
from solver.solver_result import SolverResult


class DummySolver:
    """
    Solver that returns the input angle as output angle.
    """

    def __init__(
        self,
        stage: Stage,
        *,
        precision: SolverPrecision | None = None,
    ) -> None:

        self.stage = stage
        self.precision = precision

    def solve(
        self,
        *,
        input_angle: float,
    ) -> SolverResult:

        return SolverResult(
            success=True,
            angle=input_angle,
            residual=0.0,
            iterations=1,
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


def test_motion_range_is_motion_provider():

    motion: MotionProvider = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    assert list(motion) == [
        0.0,
        math.radians(5),
        math.radians(10),
    ]


def test_adaptive_motion_range_is_motion_provider():

    motion: MotionProvider = AdaptiveMotionRange(
        start_angle=0.0,
        end_angle=math.radians(10),
    )

    angles = list(motion)

    assert len(angles) > 0

    assert math.isclose(
        angles[0],
        0.0,
    )


def test_stage_simulator_runs_with_motion_range():

    simulator = StageSimulator(
        solver_type=DummySolver,
    )

    result = simulator.run(
        stage=create_stage(),
        motion=MotionRange(
            start_angle=0.0,
            max_angle=math.radians(10),
            step=math.radians(5),
        ),
    )

    assert result.success

    assert len(result.input_angles) == 3


def test_stage_simulator_runs_with_adaptive_motion_range():

    simulator = StageSimulator(
        solver_type=DummySolver,
    )

    result = simulator.run(
        stage=create_stage(),
        motion=AdaptiveMotionRange(
            start_angle=0.0,
            end_angle=math.radians(10),
        ),
    )

    assert result.success

    assert len(result.input_angles) > 0