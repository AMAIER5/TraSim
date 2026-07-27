"""
tests/test_angle_solver.py

Unit tests for AngleSolver.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from solver.angle_solver import AngleSolver
from solver.solver_state import SolverState


def create_stage() -> Stage:
    """
    Create simple symmetric four-bar stage.
    """

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
# Reference position
# ---------------------------------------------------------------------------

def test_solver_finds_reference_position():

    stage = create_stage()

    solver = AngleSolver(
        search_window=math.radians(15),
        search_step=math.radians(0.25),
    )

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        stage,
        input_angle=0.0,
        state=state,
    )

    assert result.success

    assert math.isclose(
        result.angle,
        0.0,
        abs_tol=math.radians(0.5),
    )


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def test_solver_tracks_small_motion():

    stage = create_stage()

    solver = AngleSolver(
        search_window=math.radians(20),
        search_step=math.radians(0.25),
    )

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        stage,
        input_angle=math.radians(5),
        state=state,
    )

    assert result.success

    assert abs(result.residual) < 1e-6


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

def test_solver_reports_failure():

    stage = create_stage()

    solver = AngleSolver(
        search_window=math.radians(1),
        search_step=math.radians(0.5),
    )

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=math.pi,
    )

    result = solver.solve(
        stage,
        input_angle=math.radians(30),
        state=state,
    )

    assert result.success is False
    assert math.isnan(result.angle)