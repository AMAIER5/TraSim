"""
tests/test_coupled_lever_angle_limits.py

Regression tests for intermediate levers leaving their
admissible lever-angle segment.

Root causes covered:

1. A lever coupled to the previous stage's output lever
   was simulated in the WRONG lever-angle frame: the
   stage chain fed the previous output angles to the
   following stage 1:1, ignoring the coupling offset
   (``angle_start - coupled.angle_start``).  The coupled
   lever was therefore drawn at the previous lever's
   angle and could silently leave its own build space.

2. The stage input limits were never enforced: the
   AngleSolver only filtered OUTPUT brackets against the
   output limits, so an inadmissible input angle (an
   intermediate lever outside its segment) produced a
   "successful" solution.
"""

from __future__ import annotations

import math

import pytest

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.point_motion import PointMotion
from simulation.result_motion_provider import (
    ResultMotionProvider,
)
from simulation.simulation_result import (
    SimulationResult,
)
from simulation.stage_simulator import StageSimulator
from solver.angle_solver import AngleSolver
from solver.solver_state import SolverState


# ---------------------------------------------------------------------------
# ResultMotionProvider angle offset
# ---------------------------------------------------------------------------

def test_result_motion_provider_without_offset_passes_angles_through():
    result = SimulationResult(
        input_angles=(0.1, 0.2),
        output_angles=(0.3, 0.4),
        success=True,
    )

    provider = ResultMotionProvider(result)

    assert tuple(provider) == (0.3, 0.4)


def test_result_motion_provider_applies_angle_offset():
    result = SimulationResult(
        input_angles=(0.1, 0.2),
        output_angles=(0.3, 0.4),
        success=True,
    )

    offset = math.radians(30)
    provider = ResultMotionProvider(
        result,
        angle_offset=offset,
    )

    assert tuple(provider) == (
        0.3 + offset,
        0.4 + offset,
    )


def test_result_motion_provider_is_repeatable_with_offset():
    result = SimulationResult(
        input_angles=(0.1,),
        output_angles=(0.3,),
        success=True,
    )

    provider = ResultMotionProvider(
        result,
        angle_offset=math.radians(15),
    )

    assert tuple(provider) == tuple(provider)


# ---------------------------------------------------------------------------
# AngleSolver input limits
# ---------------------------------------------------------------------------

def create_stage(
    *,
    input_angle_min: float = float("-inf"),
    input_angle_max: float = float("inf"),
    input_angle: float = 0.0,
    output_angle: float = 0.0,
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
        input_angle=input_angle,
        output_angle=output_angle,
        input_angle_min=input_angle_min,
        input_angle_max=input_angle_max,
        validate_reference=False,
    )


def test_angle_solver_blocks_input_above_limit():
    """
    An input angle outside the stage's input limits must
    block the step instead of computing a solution that
    moves the input lever outside its build space.
    """

    stage = create_stage(
        input_angle_min=math.radians(-10),
        input_angle_max=math.radians(10),
    )
    solver = AngleSolver(stage)
    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        input_angle=math.radians(20),
        state=state,
    )

    assert result.success is False
    assert result.reason == "input_angle_limit"


def test_angle_solver_blocks_input_below_limit():
    stage = create_stage(
        input_angle_min=math.radians(-10),
        input_angle_max=math.radians(10),
    )
    solver = AngleSolver(stage)
    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        input_angle=-math.radians(20),
        state=state,
    )

    assert result.success is False
    assert result.reason == "input_angle_limit"


def test_angle_solver_accepts_input_inside_limits():
    stage = create_stage(
        input_angle_min=math.radians(-10),
        input_angle_max=math.radians(10),
    )
    solver = AngleSolver(stage)
    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        input_angle=math.radians(5),
        state=state,
    )

    assert result.success is True
    assert result.reason is None


# ---------------------------------------------------------------------------
# Coupled intermediate lever in a two-stage chain
# ---------------------------------------------------------------------------

def create_coupled_two_stage_mechanism(
    coupling_offset: float = math.radians(30),
) -> Mechanism:
    """
    Two-stage mechanism whose middle lever is coupled.

    Stage 1 drives lever A to output lever B.  Lever C is
    coupled to B with a constant angular offset (a separate
    physical lever at its own pivot, connected through a
    torsion shaft) and drives stage 2's output lever D.
    Lever C's admissible segment is [10 deg, 60 deg], so
    the coupling offset must shift B's angles into C's
    lever-angle frame to detect when C leaves its build
    space.
    """
    lever_a = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )
    lever_b = Lever(
        pivot=Point3D(100, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )
    lever_c = Lever(
        pivot=Point3D(200, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )
    lever_d = Lever(
        pivot=Point3D(300, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )

    stage1 = Stage.from_reference_position(
        input_lever=lever_a,
        output_lever=lever_b,
        input_angle=0.0,
        output_angle=0.0,
        input_angle_min=math.radians(-60),
        input_angle_max=math.radians(60),
        output_angle_min=math.radians(-60),
        output_angle_max=math.radians(60),
        validate_reference=False,
    )

    stage2 = Stage.from_reference_position(
        input_lever=lever_c,
        output_lever=lever_d,
        input_angle=coupling_offset,
        output_angle=0.0,
        input_angle_min=math.radians(10),
        input_angle_max=math.radians(60),
        output_angle_min=math.radians(-60),
        output_angle_max=math.radians(60),
        validate_reference=False,
    )

    return Mechanism(
        stages=(
            stage1,
            stage2,
        )
    )


def test_coupled_lever_receives_input_angles_in_own_frame():
    """
    The second stage's input angles must include the
    coupling offset, so they are lever angles of the
    coupled lever C and can be checked against C's own
    angle limits.
    """

    mechanism = create_coupled_two_stage_mechanism()

    results = MechanismSimulator(
        motion=PointMotion(
            angles=(
                math.radians(5),
                math.radians(10),
            ),
        ),
        stage_simulator=StageSimulator(),
    ).simulate(mechanism)

    assert results[0].success is True
    assert results[1].success is True

    coupling_offset = math.radians(30)

    for previous, coupled in zip(
        results[0].output_angles,
        results[1].input_angles,
    ):
        assert coupled == pytest.approx(
            previous + coupling_offset
        )

    for coupled in results[1].input_angles:
        assert mechanism.stages[
            1
        ].accepts_input_angle(coupled)


def test_coupled_lever_outside_segment_blocks_second_stage():
    """
    When the coupled lever C leaves its admissible segment
    [10 deg, 60 deg], the second stage must block instead of
    producing a solution that moves C outside its build
    space.  With a negative coupling offset the shifted
    angles fall below the input limit of stage 2 and every
    step blocks.
    """

    mechanism = create_coupled_two_stage_mechanism(
        coupling_offset=-math.radians(30),
    )

    results = MechanismSimulator(
        motion=PointMotion(
            angles=(
                math.radians(5),
                math.radians(10),
            ),
        ),
        stage_simulator=StageSimulator(),
    ).simulate(mechanism)

    assert results[1].success is False
    assert results[1].blocked_at is not None
