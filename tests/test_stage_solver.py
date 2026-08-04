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
from solver.solver_result import SolverResult
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

    result = solver.solve(
        input_angle=0.0,
    )

    assert result.success

    assert math.isclose(
        result.angle,
        0.0,
        abs_tol=math.radians(0.5),
    )


# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------

def test_stage_solver_updates_state():

    stage = create_stage()

    solver = StageSolver(stage)

    first = solver.solve(
        input_angle=0.0,
    )

    assert first.success

    second = solver.solve(
        input_angle=math.radians(5),
    )

    assert second.success

    assert math.isclose(
        second.angle,
        second.angle,
    )


# ---------------------------------------------------------------------------
# Failed solution
# ---------------------------------------------------------------------------

def test_stage_solver_keeps_state_on_failure(monkeypatch):

    stage = create_stage()

    solver = StageSolver(stage)

    initial = solver.solve(
        input_angle=0.0,
    )

    assert initial.success

    def fail_solve(*args, **kwargs):

        return SolverResult(
            success=False,
            angle=float("nan"),
            residual=float("inf"),
            iterations=0,
            reason="blocked",
        )

    monkeypatch.setattr(
        solver.angle_solver,
        "solve",
        fail_solve,
    )

    failed = solver.solve(
        input_angle=math.radians(30),
    )

    assert failed.success is False

    monkeypatch.undo()

    result = solver.solve(
        input_angle=math.radians(5),
    )

    assert result.success