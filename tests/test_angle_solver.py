"""
tests/test_angle_solver.py

Integration tests for the numerical AngleSolver.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from solver.angle_solver import AngleSolver
from solver.root_solver import RootSolver
from solver.solver_state import SolverState


def create_test_stage(
    *,
    input_angle_min: float = float("-inf"),
    input_angle_max: float = float("inf"),
    output_angle_min: float = float("-inf"),
    output_angle_max: float = float("inf"),
) -> Stage:

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
        input_angle_min=input_angle_min,
        input_angle_max=input_angle_max,
        output_angle_min=output_angle_min,
        output_angle_max=output_angle_max,
    )


# ---------------------------------------------------------------------------
# Basic solving
# ---------------------------------------------------------------------------


def test_angle_solver_finds_solution():
    """
    Solver returns a valid solution for a solvable geometry.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        input_angle=0.0,
        state=state,
    )

    assert result.success is True

    assert result.residual < 1e-10


# ---------------------------------------------------------------------------
# Branch continuity
# ---------------------------------------------------------------------------


def test_angle_solver_preserves_branch():
    """
    Consecutive solutions remain continuous.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
        direction=1,
    )

    previous = state.last_output_angle

    for input_angle in (
        math.radians(5),
        math.radians(10),
        math.radians(15),
    ):

        result = solver.solve(
            input_angle=input_angle,
            state=state,
        )

        assert result.success is True

        assert abs(
            result.angle - previous
        ) < math.radians(20)

        state = state.next(
            input_angle=input_angle,
            output_angle=result.angle,
        )

        previous = result.angle


# ---------------------------------------------------------------------------
# Kinematic branch selection
# ---------------------------------------------------------------------------


def test_angle_solver_prefers_continuous_velocity_branch():
    """
    Branch selection prefers kinematic continuity over
    a closer prediction if the velocity jump becomes
    unrealistic.
    """

    brackets = [
        (
            math.radians(-23),
            math.radians(-21),
            1,
        ),
        (
            math.radians(-36),
            math.radians(-34),
            1,
        ),
    ]

    state = SolverState(
        last_input_angle=math.radians(10),
        last_output_angle=math.radians(-10),
        direction=-1,
        output_velocity=-5.0,
    )

    selected = AngleSolver._select_branch(
        brackets,
        reference_angle=math.radians(-22),
        state=state,
        input_angle=math.radians(15),
    )

    left, right, _ = selected

    selected_angle = (
        left + right
    ) / 2.0

    assert math.isclose(
        selected_angle,
        math.radians(-35),
        abs_tol=math.radians(1),
    )


def test_angle_solver_without_motion_history_uses_prediction():
    """
    Initial solver state behaves like the previous
    prediction-based branch selection.
    """

    brackets = [
        (
            math.radians(-31),
            math.radians(-29),
            1,
        ),
        (
            math.radians(20),
            math.radians(22),
            1,
        ),
    ]

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
        direction=0,
        output_velocity=0.0,
    )

    selected = AngleSolver._select_branch(
        brackets,
        reference_angle=math.radians(-30),
        state=state,
        input_angle=math.radians(5),
    )

    left, right, _ = selected

    selected_angle = (
        left + right
    ) / 2.0

    assert math.isclose(
        selected_angle,
        math.radians(-30),
        abs_tol=math.radians(1),
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_angle_solver_iteration_limit():
    """
    Numerical solver should converge quickly.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        input_angle=0.0,
        state=state,
    )

    assert result.success is True

    assert result.success is True

    assert solver.get_stats()["solved"] == 1

# ---------------------------------------------------------------------------
# Output angle limits
# ---------------------------------------------------------------------------


def test_angle_solver_global_search_respects_stage_output_limits(
    monkeypatch,
):
    """
    Global bracket search is restricted to the Stage
    output angle limits.
    """

    stage = create_test_stage(
        output_angle_min=math.radians(-20),
        output_angle_max=math.radians(35),
    )

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    captured = {}

    def fake_find_all_brackets(
        *,
        function,
        minimum,
        maximum,
        step,
    ):
        captured["minimum"] = minimum
        captured["maximum"] = maximum
        return []

    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets",
        fake_find_all_brackets,
    )

    result = solver.solve(
        input_angle=math.radians(90),
        state=state,
    )

    assert result.success is False

    assert captured["minimum"] == math.radians(-20)

    assert captured["maximum"] == math.radians(35)