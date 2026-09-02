"""
tests/test_stage.py

Unit tests for mechanical Stage component.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from tests.test_angle_solver import create_test_stage

# ---------------------------------------------------------------------------
# Basic creation
# ---------------------------------------------------------------------------

def test_stage_creation():

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

    stage = Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=0.0,
        output_angle=0.0,
    )

    assert stage.input_lever == input_lever
    assert stage.output_lever == output_lever


# ---------------------------------------------------------------------------
# Rod length generation
# ---------------------------------------------------------------------------

def test_stage_calculates_rod_length():

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

    stage = Stage.from_reference_position(
        input_lever,
        output_lever,
        0.0,
        0.0,
    )

    assert math.isclose(
        stage.rod_length,
        0.0 + 100.0,
    )


# ---------------------------------------------------------------------------
# Reference endpoints
# ---------------------------------------------------------------------------

def test_reference_endpoints():

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

    stage = Stage.from_reference_position(
        input_lever,
        output_lever,
        0.0,
        0.0,
    )

    assert stage.input_endpoint.almost_equal(
        Point3D(50, 0, 0)
    )

    assert stage.output_endpoint.almost_equal(
        Point3D(150, 0, 0)
    )


# ---------------------------------------------------------------------------
# Endpoint calculation
# ---------------------------------------------------------------------------

def test_stage_endpoint_for_angle():

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

    stage = Stage.from_reference_position(
        input_lever,
        output_lever,
        0.0,
        0.0,
    )

    point = stage.input_position(
        math.pi / 2
    )

    assert point.almost_equal(
        Point3D(0, 50, 0)
    )
    
def test_stage_accepts_input_angle_inside_range():

    stage = create_test_stage(
        input_angle_min=math.radians(-30),
        input_angle_max=math.radians(30),
    )

    assert stage.accepts_input_angle(
        math.radians(10)
    ) is True


def test_stage_rejects_output_angle_outside_range():

    stage = create_test_stage(
        output_angle_min=math.radians(-20),
        output_angle_max=math.radians(20),
    )

    assert stage.accepts_output_angle(
        math.radians(30)
    ) is False