"""
tests/test_stage_simulator.py
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from simulation.motion_range import MotionRange
from simulation.stage_simulator import StageSimulator
from solver.solver_result import SolverResult
from solver.solver_state import SolverState


class DummySolver:
    """
    Solver that always succeeds.
    """

    def __init__(
        self,
        stage: Stage,
    ) -> None:

        self.stage = stage

    def solve(
        self,
        *,
        input_angle: float,
        state: SolverState,
    ):

        return (
            SolverResult(
                success=True,
                angle=input_angle,
                residual=0.0,
                iterations=1,
            ),
            state,
        )


class BlockingSolver:
    """
    Blocks during second solver call.
    """

    def __init__(
        self,
        stage: Stage,
    ) -> None:

        self.calls = 0

    def solve(
        self,
        *,
        input_angle: float,
        state: SolverState,
    ):

        self.calls += 1

        if self.calls == 2:

            return (
                SolverResult(
                    success=False,
                    angle=float("nan"),
                    residual=1.0,
                    iterations=1,
                    reason="Blocked",
                ),
                state,
            )

        return (
            SolverResult(
                success=True,
                angle=input_angle,
                residual=0.0,
                iterations=1,
            ),
            state,
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


def create_motion() -> MotionRange:

    return MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )


def test_successful_simulation():

    simulator = StageSimulator(
        solver_type=DummySolver,
    )

    result = simulator.run(
        stage=create_stage(),
        motion=create_motion(),
    )

    assert result.success

    assert result.input_angles == (
        0.0,
        math.radians(5),
        math.radians(10),
    )

    assert result.output_angles == (
        0.0,
        math.radians(5),
        math.radians(10),
    )


def test_blocking_simulation():

    simulator = StageSimulator(
        solver_type=BlockingSolver,
    )

    result = simulator.run(
        stage=create_stage(),
        motion=create_motion(),
    )

    assert not result.success

    assert math.isclose(
        result.blocked_at,
        math.radians(5),
    )


def test_simulator_is_reusable():

    simulator = StageSimulator(
        solver_type=DummySolver,
    )

    motion = create_motion()

    first = simulator.run(
        stage=create_stage(),
        motion=motion,
    )

    second = simulator.run(
        stage=create_stage(),
        motion=motion,
    )

    assert first.success
    assert second.success

    assert (
        first.input_angles
        == second.input_angles
    )

    assert (
        first.output_angles
        == second.output_angles
    )