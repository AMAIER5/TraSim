"""
tests/test_bidirectional_simulator.py

Tests for bidirectional stage simulation.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.lever import Lever
from mechanics.stage import Stage

from simulation.bidirectional_simulator import (
    BidirectionalSimulator,
)


def create_stage() -> Stage:
    """
    Create simple symmetric stage.
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
# Basic bidirectional simulation
# ---------------------------------------------------------------------------

def test_bidirectional_simulation_runs():

    stage = create_stage()

    simulator = BidirectionalSimulator(stage)

    result = simulator.run(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    assert result.success

    assert len(result.input_angles) > 0

    assert len(result.input_angles) == len(
        result.output_angles
    )


# ---------------------------------------------------------------------------
# Contains both directions
# ---------------------------------------------------------------------------

def test_bidirectional_contains_negative_and_positive_angles():

    stage = create_stage()

    simulator = BidirectionalSimulator(stage)

    result = simulator.run(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    assert any(
        angle < 0
        for angle in result.input_angles
    )

    assert any(
        angle > 0
        for angle in result.input_angles
    )


# ---------------------------------------------------------------------------
# Start angle is included
# ---------------------------------------------------------------------------

def test_bidirectional_contains_start_angle():

    stage = create_stage()

    simulator = BidirectionalSimulator(stage)

    result = simulator.run(
        start_angle=math.radians(15),
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    assert math.isclose(
        result.input_angles[0],
        math.radians(15),
    )
    
def test_bidirectional_order_starts_at_start_angle():

    stage = create_stage()

    simulator = BidirectionalSimulator(stage)

    result = simulator.run(
        start_angle=math.radians(15),
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    expected = [
        math.radians(15),
        math.radians(10),
        math.radians(5),
        math.radians(20),
        math.radians(25),
    ]

    assert len(result.input_angles) == len(expected)

    for actual, reference in zip(
        result.input_angles,
        expected,
    ):
        assert math.isclose(
            actual,
            reference,
        )