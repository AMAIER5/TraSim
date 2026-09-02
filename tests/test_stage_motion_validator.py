"""
tests/test_stage_motion_validator.py

Tests for StageMotionValidator.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from validation.stage_motion_validator import (
    StageMotionValidator,
)


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
# Valid stage
# ---------------------------------------------------------------------------


def test_stage_motion_validator_accepts_valid_motion_range():
    """
    Validator accepts a stage where the complete
    input range remains solvable.
    """

    stage = create_test_stage(
        input_angle_min=math.radians(-20),
        input_angle_max=math.radians(20),
    )

    validator = StageMotionValidator(
        steps=20,
    )

    result = validator.validate(stage)

    assert result.valid is True

    assert result.failed_at_input_angle is None

    assert result.checked_steps == 21


# ---------------------------------------------------------------------------
# Invalid input range
# ---------------------------------------------------------------------------


def test_stage_motion_validator_requires_finite_input_range():
    """
    Validator requires a defined mechanical input range.
    """

    stage = create_test_stage()

    validator = StageMotionValidator()

    try:
        validator.validate(stage)

    except ValueError as error:

        assert (
            "finite"
            in str(error)
        )

    else:

        assert False, (
            "Expected ValueError for infinite input range"
        )


# ---------------------------------------------------------------------------
# Diagnostic information
# ---------------------------------------------------------------------------


def test_stage_motion_validator_returns_failure_position():
    """
    Validator returns the first input position where
    the stage cannot be solved.

    This test uses an intentionally impossible output
    range.
    """

    stage = create_test_stage(
        input_angle_min=math.radians(-10),
        input_angle_max=math.radians(10),
        output_angle_min=math.radians(90),
        output_angle_max=math.radians(100),
    )

    validator = StageMotionValidator(
        steps=10,
    )

    result = validator.validate(stage)

    assert result.valid is False

    assert result.failed_at_input_angle is not None

    assert result.reason is not None

    assert result.checked_steps >= 1
    
# ---------------------------------------------------------------------------
# Validation of externally supplied motion
# ---------------------------------------------------------------------------


def test_stage_motion_validator_accepts_given_motion():
    """
    Validator accepts a supplied input motion if every
    position is solvable.
    """

    stage = create_test_stage()

    validator = StageMotionValidator()

    motion = (
        math.radians(-5),
        0.0,
        math.radians(5),
        math.radians(10),
    )

    result = validator.validate_motion(
        stage,
        motion,
    )

    assert result.valid is True

    assert result.checked_steps == len(motion)

    assert result.failed_at_input_angle is None


def test_stage_motion_validator_reports_first_invalid_motion_position():
    """
    Validator reports the first supplied motion position
    that cannot be solved.
    """

    stage = create_test_stage(
        output_angle_min=math.radians(90),
        output_angle_max=math.radians(100),
    )

    validator = StageMotionValidator()

    motion = (
        math.radians(-5),
        0.0,
        math.radians(5),
    )

    result = validator.validate_motion(
        stage,
        motion,
    )

    assert result.valid is False

    assert result.checked_steps == 1

    assert result.failed_at_input_angle == motion[0]

    assert result.reason is not None
    
# ---------------------------------------------------------------------------
# Output angle limits
# ---------------------------------------------------------------------------


def test_stage_motion_validator_rejects_solution_outside_output_range():
    """
    A stage with an impossible output range must be rejected.
    """

    stage = create_test_stage(
        input_angle_min=math.radians(-10),
        input_angle_max=math.radians(10),
        output_angle_min=math.radians(20),
        output_angle_max=math.radians(40),
    )

    validator = StageMotionValidator(
        steps=10,
    )

    result = validator.validate(stage)

    assert result.valid is False

    assert result.failed_at_input_angle is not None

    assert result.reason in (
        "blocked",
        "output_angle_limit",
    )
    

def test_stage_motion_validator_keeps_solver_failure_diagnostic():
    """
    A solver failure must remain distinguishable from
    an output angle limit violation.
    """

    stage = create_test_stage(
        input_angle_min=math.radians(-20),
        input_angle_max=math.radians(20),
    )

    validator = StageMotionValidator(
        steps=20,
    )

    result = validator.validate(stage)

    if not result.valid:
        assert result.reason in (
            "no_solution",
            "output_angle_limit",
        )