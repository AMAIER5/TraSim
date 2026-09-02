"""
tests/test_mechanism_motion_validator.py
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