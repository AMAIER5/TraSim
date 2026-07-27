"""
tests/test_stage_solver.py

Unit tests for StageSolver.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from solver.solver_state import SolverState
from solver.stage_solver import StageSolver


def create_stage() -> Stage:
    """
    Create simple symmetric stage.
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
# Creation
# ---------------------------------------------------------------------------

def test_stage_solver_creation():

    stage = create_stage()

    solver = StageSolver(stage)

    assert solver.stage == stage


# ---------------------------------------------------------------------------
# First solution
# ---------------------------------------------------------------------------

def test_stage_solver_solves_reference_position():

    stage = create_stage()

    solver = StageSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result, new_state = solver.solve(
        input_angle=0.0,
        state=state,
    )

    assert result.success

    assert math.isclose(
        result.angle,
        0.0,
        abs_tol=math.radians(0.5),
    )

    assert new_state.last_input_angle == 0.0


# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------

def test_stage_solver_updates_state():

    stage = create_stage()

    solver = StageSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result, new_state = solver.solve(
        input_angle=math.radians(5),
        state=state,
    )

    assert result.success

    assert math.isclose(
        new_state.last_input_angle,
        math.radians(5),
    )

    assert math.isclose(
        new_state.last_output_angle,
        result.angle,
    )


# ---------------------------------------------------------------------------
# Failed solution
# ---------------------------------------------------------------------------

def test_stage_solver_keeps_state_on_failure():

    stage = create_stage()

    solver = StageSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=math.pi,
    )

    result, new_state = solver.solve(
        input_angle=math.radians(30),
        state=state,
    )

    assert result.success is False

    assert new_state == state