"""
tests/test_stage.py

Unit tests for mechanical Stage component.
"""

from __future__ import annotations

import math

import pytest

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


# ---------------------------------------------------------------------------
# Issue #7: Reference angle validation
# ---------------------------------------------------------------------------

def _make_levers():
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
    return input_lever, output_lever


def test_reference_inside_range_succeeds():
    """
    Issue #7: A reference position inside the declared
    ranges must be accepted.
    """

    input_lever, output_lever = _make_levers()

    stage = Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=math.radians(5),
        output_angle=math.radians(-3),
        input_angle_min=math.radians(-30),
        input_angle_max=math.radians(30),
        output_angle_min=math.radians(-20),
        output_angle_max=math.radians(20),
    )

    assert stage.input_angle == math.radians(5)
    assert stage.output_angle == math.radians(-3)


def test_reference_input_above_max_raises():
    """
    Issue #7: An input reference above input_angle_max
    must raise ValueError.
    """

    input_lever, output_lever = _make_levers()

    with pytest.raises(ValueError, match="input angle"):
        Stage.from_reference_position(
            input_lever=input_lever,
            output_lever=output_lever,
            input_angle=math.radians(40),
            input_angle_min=math.radians(-30),
            input_angle_max=math.radians(30),
        )


def test_reference_input_below_min_raises():
    """
    Issue #7: An input reference below input_angle_min
    must raise ValueError.
    """

    input_lever, output_lever = _make_levers()

    with pytest.raises(ValueError, match="input angle"):
        Stage.from_reference_position(
            input_lever=input_lever,
            output_lever=output_lever,
            input_angle=math.radians(-40),
            input_angle_min=math.radians(-30),
            input_angle_max=math.radians(30),
        )


def test_reference_output_above_max_raises():
    """
    Issue #7: An output reference above output_angle_max
    must raise ValueError.
    """

    input_lever, output_lever = _make_levers()

    with pytest.raises(ValueError, match="output angle"):
        Stage.from_reference_position(
            input_lever=input_lever,
            output_lever=output_lever,
            output_angle=math.radians(25),
            output_angle_min=math.radians(-20),
            output_angle_max=math.radians(20),
        )


def test_reference_output_below_min_raises():
    """
    Issue #7: An output reference below output_angle_min
    must raise ValueError.
    """

    input_lever, output_lever = _make_levers()

    with pytest.raises(ValueError, match="output angle"):
        Stage.from_reference_position(
            input_lever=input_lever,
            output_lever=output_lever,
            output_angle=math.radians(-25),
            output_angle_min=math.radians(-20),
            output_angle_max=math.radians(20),
        )


def test_reference_at_boundary_succeeds():
    """
    Issue #7: A reference exactly at the boundary
    (min or max) must be accepted.
    """

    input_lever, output_lever = _make_levers()

    # At min boundary.
    stage_min = Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=math.radians(-30),
        output_angle=math.radians(-20),
        input_angle_min=math.radians(-30),
        input_angle_max=math.radians(30),
        output_angle_min=math.radians(-20),
        output_angle_max=math.radians(20),
    )
    assert stage_min.input_angle == math.radians(-30)

    # At max boundary.
    stage_max = Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=math.radians(30),
        output_angle=math.radians(20),
        input_angle_min=math.radians(-30),
        input_angle_max=math.radians(30),
        output_angle_min=math.radians(-20),
        output_angle_max=math.radians(20),
    )
    assert stage_max.input_angle == math.radians(30)


def test_reference_no_range_constraint_succeeds():
    """
    Issue #7: When no ranges are supplied (default ±inf),
    any reference must be accepted.
    """

    input_lever, output_lever = _make_levers()

    stage = Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=math.radians(999),
        output_angle=math.radians(-999),
    )

    assert stage.input_angle == math.radians(999)
    assert stage.output_angle == math.radians(-999)


def test_validate_reference_false_allows_impossible_stage():
    """
    Issue #7: Setting validate_reference=False allows
    constructing a stage whose reference lies outside
    the declared ranges.  This is needed by tests that
    intentionally create impossible stages to verify
    the motion validator's rejection behavior.
    """

    input_lever, output_lever = _make_levers()

    # This would normally raise, but with
    # validate_reference=False it is allowed.
    stage = Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=0.0,
        output_angle=0.0,
        input_angle_min=math.radians(-10),
        input_angle_max=math.radians(10),
        output_angle_min=math.radians(90),
        output_angle_max=math.radians(100),
        validate_reference=False,
    )

    assert stage.output_angle == 0.0
    assert stage.output_angle_min == math.radians(90)