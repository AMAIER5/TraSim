"""
tests/test_angle_solver.py

Integration tests for the numerical AngleSolver.
"""

from __future__ import annotations

import math

import pytest

from core.point3d import Point3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from solver.angle_solver import AngleSolver
from solver.solver_state import SolverState
from core.vector3d import Vector3D

def create_test_stage() -> Stage:
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


def test_angle_solver_finds_solution():
    """
    Solver returns a valid solution for a solvable geometry.
    """

    stage = create_test_stage()

    solver = AngleSolver()

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        stage=stage,
        input_angle=0.0,
        state=state,
    )

    assert result.success is True

    assert result.residual < 1e-10


def test_angle_solver_preserves_branch():
    """
    Consecutive solutions remain continuous.
    """

    stage = create_test_stage()

    solver = AngleSolver(
        search_window=math.radians(20),
        bracket_step=math.radians(1),
    )

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
            stage=stage,
            input_angle=input_angle,
            state=state,
        )

        assert result.success is True

        assert abs(
            result.angle - previous
        ) < math.radians(20)

        previous = result.angle

        state = SolverState(
            last_input_angle=input_angle,
            last_output_angle=result.angle,
            direction=state.direction,
        )


def test_angle_solver_iteration_limit():
    """
    Numerical solver should converge quickly.
    """

    stage = create_test_stage()

    solver = AngleSolver()

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        stage=stage,
        input_angle=0.0,
        state=state,
    )

    assert result.success is True

    assert result.iterations < 60
