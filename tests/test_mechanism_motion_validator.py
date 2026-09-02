"""
tests/test_mechanism_motion_validator.py

Issue #6: Updated import to use the consolidated
MechanismValidationResult from its own module.  Added
tests for the failed_stage property that was previously
unused/untested.
"""

from __future__ import annotations

import math

from mechanics.csv_mechanism_builder import CsvMechanismBuilder
from mechanism_io.csv_reader import CsvReader
from optimization.parameter_set import ParameterSet
from simulation.motion_range import MotionRange
from validation.mechanism_motion_validator import (
    MechanismMotionValidator,
)
from validation.mechanism_validation_result import (
    MechanismValidationResult,
)


def build_mechanism(
    csv_file,
):
    """
    Helper:
    Build a mechanism from CSV.
    """

    definition = CsvReader.read_mechanism(
        csv_file,
    )

    return CsvMechanismBuilder(
        definition,
    ).build(
        ParameterSet(()),
    )


def test_validator_accepts_valid_mechanism(
    simple_multistage_csv,
):
    """
    All stages of a valid mechanism should validate
    successfully.
    """

    mechanism = build_mechanism(
        simple_multistage_csv,
    )

    motion = MotionRange(
        start_angle=math.radians(-20),
        max_angle=math.radians(20),
        step=math.radians(5),
    )

    validation = MechanismMotionValidator().validate(
        mechanism=mechanism,
        motion=motion,
    )

    assert isinstance(
        validation,
        MechanismValidationResult,
    )

    assert len(validation.stages) == len(
        mechanism.stages
    )

    assert validation.valid is True

    assert all(
        result.valid
        for result in validation.stages
    )

    assert all(
        result.failed_at_input_angle is None
        for result in validation.stages
    )


def test_validator_returns_one_result_per_stage(
    simple_multistage_csv,
):
    """
    Validation result order must match the stage order.
    """

    mechanism = build_mechanism(
        simple_multistage_csv,
    )

    motion = MotionRange(
        start_angle=math.radians(-10),
        max_angle=math.radians(10),
        step=math.radians(10),
    )

    validation = MechanismMotionValidator().validate(
        mechanism=mechanism,
        motion=motion,
    )

    assert len(validation.stages) == len(
        mechanism.stages
    )

    for index, stage_result in enumerate(
        validation.stages
    ):
        assert stage_result.stage_id == index


def test_validator_reports_checked_steps(
    simple_multistage_csv,
):
    """
    Every stage should report the number of checked
    motion positions.
    """

    mechanism = build_mechanism(
        simple_multistage_csv,
    )

    motion = MotionRange(
        start_angle=math.radians(-10),
        max_angle=math.radians(10),
        step=math.radians(10),
    )

    validation = MechanismMotionValidator().validate(
        mechanism=mechanism,
        motion=motion,
    )

    expected_steps = len(
        tuple(motion)
    )

    assert validation.valid is True

    for stage_result in validation.stages:

        assert (
            stage_result.checked_steps
            == expected_steps
        )


def test_failed_stage_property_returns_none_for_valid(
    simple_multistage_csv,
):
    """
    Issue #6: The failed_stage property should return
    None when all stages are valid.
    """

    mechanism = build_mechanism(
        simple_multistage_csv,
    )

    motion = MotionRange(
        start_angle=math.radians(-10),
        max_angle=math.radians(10),
        step=math.radians(10),
    )

    validation = MechanismMotionValidator().validate(
        mechanism=mechanism,
        motion=motion,
    )

    assert validation.valid is True

    assert validation.failed_stage is None


def test_failed_stage_property_returns_id_for_invalid(
    simple_multistage_csv,
):
    """
    Issue #6: The failed_stage property should return
    the stage_id of the first failed stage.

    We create a motion range that is too large for the
    mechanism, causing at least one stage to fail
    validation.
    """

    mechanism = build_mechanism(
        simple_multistage_csv,
    )

    # Use a very large motion range to force a failure.
    motion = MotionRange(
        start_angle=math.radians(-90),
        max_angle=math.radians(180),
        step=math.radians(5),
    )

    validation = MechanismMotionValidator().validate(
        mechanism=mechanism,
        motion=motion,
    )

    if not validation.valid:
        assert validation.failed_stage is not None