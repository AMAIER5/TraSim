"""
tests/test_single_stage_simulation.py

End-to-end test for single stage simulation.
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
    Create simple symmetric four-bar mechanism.
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
# Full movement simulation
# ---------------------------------------------------------------------------

def test_single_stage_simulation():

    stage = create_stage()

    solver = StageSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    input_angles = [
        math.radians(angle)
        for angle in range(0, 31, 5)
    ]

    output_angles = []

    for input_angle in input_angles:

        result, state = solver.solve(
            input_angle=input_angle,
            state=state,
        )

        assert result.success

        output_angles.append(
            result.angle
        )

    assert len(output_angles) == len(input_angles)


# ---------------------------------------------------------------------------
# Continuity check
# ---------------------------------------------------------------------------

def test_single_stage_motion_is_continuous():

    stage = create_stage()

    solver = StageSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    previous_angle = None

    for input_deg in range(0, 31, 2):

        result, state = solver.solve(
            input_angle=math.radians(input_deg),
            state=state,
        )

        assert result.success

        if previous_angle is not None:

            delta = abs(
                result.angle - previous_angle
            )

            assert delta < math.radians(20)

        previous_angle = result.angle


# ---------------------------------------------------------------------------
# State follows simulation
# ---------------------------------------------------------------------------

def test_simulation_state_matches_last_result():

    stage = create_stage()

    solver = StageSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result, state = solver.solve(
        input_angle=math.radians(10),
        state=state,
    )

    assert result.success

    assert math.isclose(
        state.last_input_angle,
        math.radians(10),
    )

    assert math.isclose(
        state.last_output_angle,
        result.angle,
    )