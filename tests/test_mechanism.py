"""
tests/test_mechanism.py

Tests for Mechanism container.
"""

from __future__ import annotations

import pytest

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage


def create_stage() -> Stage:
    """
    Create minimal valid stage.
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


def test_empty_mechanism_is_allowed():

    mechanism = Mechanism(
        stages=()
    )

    assert len(mechanism.stages) == 0


def test_mechanism_stores_stages():

    stage_a = create_stage()
    stage_b = create_stage()

    mechanism = Mechanism(
        stages=(
            stage_a,
            stage_b,
        )
    )

    assert len(mechanism.stages) == 2

    assert mechanism.stages[0] == stage_a
    assert mechanism.stages[1] == stage_b


def test_mechanism_is_immutable():

    mechanism = Mechanism(
        stages=()
    )

    with pytest.raises(
        AttributeError
    ):
        mechanism.stages = ()


def test_mechanism_requires_tuple():

    with pytest.raises(
        TypeError
    ):
        Mechanism(
            stages=[]
        )