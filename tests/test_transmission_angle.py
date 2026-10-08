"""tests/test_transmission_angle.py

Tests for the soft minimum transmission angle criterion:
- stage transmission angle geometry (0, 90 deg, supplement)
- minimum over a mechanism
- soft penalty behaviour of TransmissionAngleConstraint
"""

from __future__ import annotations

import math

import pytest

from analysis.transmission_angle import (
    MIN_TRANSMISSION_ANGLE,
    TransmissionAngleConstraint,
    min_transmission_angle,
    stage_transmission_angle,
)
from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage


def _stage(
    input_end: tuple[float, float, float],
    output_end: tuple[float, float, float],
) -> Stage:
    input_lever = Lever(
        pivot=Point3D(0.0, 0.0, 0.0),
        axis=Vector3D(0.0, 0.0, 1.0),
        length=1.0,
    )
    output_pivot = Point3D(10.0, 0.0, 0.0)
    output_lever = Lever(
        pivot=output_pivot,
        axis=Vector3D(0.0, 0.0, 1.0),
        length=1.0,
    )
    rod = (
        Point3D(*output_end)
        - Point3D(*input_end)
    ).norm()
    return Stage(
        input_lever=input_lever,
        output_lever=output_lever,
        rod_length=rod,
        input_angle_min=-1.0,
        input_angle_max=1.0,
        output_angle_min=-1.0,
        output_angle_max=1.0,
        input_angle=0.0,
        output_angle=0.0,
        input_endpoint=Point3D(*input_end),
        output_endpoint=Point3D(*output_end),
    )


def test_perpendicular_rod_is_90_deg():
    stage = _stage(
        (1.0, 1.0, 0.0),
        (10.0, 1.0, 0.0),
    )
    angle = stage_transmission_angle(stage)
    assert angle == pytest.approx(
        math.pi / 2
    )


def test_collinear_rod_is_zero_deg():
    stage = _stage(
        (1.0, 0.0, 0.0),
        (11.0, 0.0, 0.0),
    )
    angle = stage_transmission_angle(stage)
    assert angle == pytest.approx(0.0)


def test_obtuse_angle_uses_supplement():
    stage = _stage(
        (1.0, 0.0, 0.0),
        (10.0, 1.0, 0.0),
    )
    angle = stage_transmission_angle(stage)
    assert angle <= math.pi / 2 + 1e-12
    expected = math.acos(1.0 / math.sqrt(82.0))
    assert angle == pytest.approx(expected)


def _example_mechanism(beta_deg=180.0):
    from pathlib import Path

    from mechanism_io.csv_reader import (
        CsvReader,
    )
    from mechanics.mixed_brake_builder import (
        MixedBrakeBuilder,
    )

    definition = CsvReader.read_mechanism(
        Path(__file__).parent.parent
        / "examples"
        / "mixed_brake_mechanism.csv",
    )
    return MixedBrakeBuilder(
        definition
    ).build(math.radians(beta_deg))


def test_min_over_mechanism():
    mechanism = _example_mechanism()
    minimum = min_transmission_angle(mechanism)
    per_stage = [
        stage_transmission_angle(stage)
        for stage in mechanism.stages
    ]
    assert minimum == pytest.approx(
        min(per_stage)
    )


def test_constraint_zero_when_satisfied():
    constraint = TransmissionAngleConstraint(
        minimum_angle=0.1,
        weight=2.0,
    )
    mechanism = _example_mechanism()
    penalty = constraint.evaluate([mechanism])
    assert penalty == 0.0


def test_constraint_penalty_quadratic():
    mechanism = _example_mechanism()
    observed = min_transmission_angle(
        mechanism
    )
    desired = observed + 0.2
    constraint = TransmissionAngleConstraint(
        minimum_angle=desired,
        weight=3.0,
    )
    penalty = constraint.evaluate([mechanism])
    deviation = 0.2
    assert penalty == pytest.approx(
        3.0 * deviation * deviation
    )


def test_constraint_empty_is_zero():
    constraint = TransmissionAngleConstraint()
    assert constraint.evaluate([]) == 0.0


def test_constraint_rejects_negative_inputs():
    with pytest.raises(ValueError):
        TransmissionAngleConstraint(
            minimum_angle=-0.1,
        )
    with pytest.raises(ValueError):
        TransmissionAngleConstraint(
            weight=-1.0,
        )


def test_default_minimum_angle():
    constraint = TransmissionAngleConstraint()
    assert constraint.minimum_angle == (
        MIN_TRANSMISSION_ANGLE
    )
    assert constraint.weight == 1.0
