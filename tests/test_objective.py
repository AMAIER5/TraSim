"""
tests/test_objective.py

Unit tests for stage objective function.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from solver.objective import stage_error


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


# ---------------------------------------------------------------------------
# Reference position
# ---------------------------------------------------------------------------

def test_stage_error_reference_position():

    stage = create_stage()

    assert math.isclose(
        stage_error(
            stage,
            0.0,
            0.0,
        ),
        0.0,
    )


# ---------------------------------------------------------------------------
# Wrong output angle
# ---------------------------------------------------------------------------

def test_stage_error_changes_with_output_angle():

    stage = create_stage()

    error = stage_error(
        stage,
        0.0,
        math.radians(10),
    )

    assert abs(error) > 1e-6


# ---------------------------------------------------------------------------
# Wrong input angle
# ---------------------------------------------------------------------------

def test_stage_error_changes_with_input_angle():

    stage = create_stage()

    error = stage_error(
        stage,
        math.radians(10),
        0.0,
    )

    assert abs(error) > 1e-6


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def test_stage_error_is_repeatable():

    stage = create_stage()

    e1 = stage_error(
        stage,
        math.radians(15),
        math.radians(5),
    )

    e2 = stage_error(
        stage,
        math.radians(15),
        math.radians(5),
    )

    assert math.isclose(e1, e2)