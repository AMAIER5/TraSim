"""
tests/test_adaptive_motion_range.py

Unit tests for AdaptiveMotionRange.
"""

from __future__ import annotations

import math

import pytest

from simulation.adaptive_motion_range import AdaptiveMotionRange


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------

def test_adaptive_motion_range_generates_angles():

    motion = AdaptiveMotionRange(
        start_angle=0.0,
        end_angle=math.radians(10),
        initial_step=math.radians(5),
    )

    angles = list(motion)

    assert len(angles) == 3

    assert math.isclose(
        angles[0],
        0.0,
    )

    assert math.isclose(
        angles[-1],
        math.radians(10),
    )


# ---------------------------------------------------------------------------
# Step reduction
# ---------------------------------------------------------------------------

def test_adaptive_motion_range_reduces_step():

    motion = AdaptiveMotionRange(
        start_angle=0.0,
        end_angle=1.0,
        initial_step=1.0,
        min_step=0.1,
        max_step=2.0,
        max_output_delta=0.5,
    )

    motion.feedback(
        output_delta=1.0,
    )

    assert math.isclose(
        motion.current_step,
        0.5,
    )


# ---------------------------------------------------------------------------
# Step increase
# ---------------------------------------------------------------------------

def test_adaptive_motion_range_increases_step():

    motion = AdaptiveMotionRange(
        start_angle=0.0,
        end_angle=1.0,
        initial_step=1.0,
        min_step=0.1,
        max_step=2.0,
        max_output_delta=1.0,
    )

    motion.feedback(
        output_delta=0.1,
    )

    assert math.isclose(
        motion.current_step,
        1.5,
    )


# ---------------------------------------------------------------------------
# Step limits
# ---------------------------------------------------------------------------

def test_adaptive_motion_range_respects_min_step():

    motion = AdaptiveMotionRange(
        start_angle=0.0,
        end_angle=1.0,
        initial_step=0.2,
        min_step=0.2,
        max_step=1.0,
        max_output_delta=0.1,
    )

    motion.feedback(
        output_delta=10.0,
    )

    assert math.isclose(
        motion.current_step,
        0.2,
    )


def test_adaptive_motion_range_respects_max_step():

    motion = AdaptiveMotionRange(
        start_angle=0.0,
        end_angle=1.0,
        initial_step=0.5,
        min_step=0.1,
        max_step=0.6,
        max_output_delta=1.0,
    )

    motion.feedback(
        output_delta=0.0,
    )

    assert math.isclose(
        motion.current_step,
        0.6,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_adaptive_motion_range_invalid_initial_step():

    with pytest.raises(ValueError):

        AdaptiveMotionRange(
            start_angle=0.0,
            end_angle=1.0,
            initial_step=0.0,
        )


def test_adaptive_motion_range_invalid_step_limits():

    with pytest.raises(ValueError):

        AdaptiveMotionRange(
            start_angle=0.0,
            end_angle=1.0,
            min_step=2.0,
            max_step=1.0,
        )