"""
tests/test_motion_range.py

Unit tests for MotionRange.
"""

from __future__ import annotations

import math

import pytest

from simulation.motion_range import MotionRange


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------

def test_motion_range_positive_direction():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(2),
    )

    angles = list(motion)

    assert len(angles) == 6

    assert math.isclose(
        angles[0],
        0.0,
    )

    assert math.isclose(
        angles[-1],
        math.radians(10),
    )


# ---------------------------------------------------------------------------
# Negative direction
# ---------------------------------------------------------------------------

def test_motion_range_negative_direction():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(2),
        direction=-1,
    )

    angles = list(motion)

    assert len(angles) == 6

    assert math.isclose(
        angles[1],
        math.radians(-2),
    )


# ---------------------------------------------------------------------------
# Invalid values
# ---------------------------------------------------------------------------

def test_motion_range_invalid_direction():

    with pytest.raises(ValueError):

        MotionRange(
            start_angle=0.0,
            max_angle=1.0,
            step=0.1,
            direction=0,
        )


def test_motion_range_invalid_step():

    with pytest.raises(ValueError):

        MotionRange(
            start_angle=0.0,
            max_angle=1.0,
            step=0.0,
        )


# ---------------------------------------------------------------------------
# Partial step
# ---------------------------------------------------------------------------

def test_motion_range_stops_at_limit():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=1.0,
        step=0.3,
    )

    angles = list(motion)

    assert math.isclose(
        angles[-1],
        0.9,
    )