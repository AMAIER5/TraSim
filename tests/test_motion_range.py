"""
tests/test_motion_range.py

Unit tests for MotionRange.

Issue #12: Added tests for count property, single-point
motion, floating-point stability, and exact count
expectations to document the iteration semantics.
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


def test_motion_range_feedback_is_ignored():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=1.0,
        step=0.1,
    )

    motion.feedback(
        output_delta=0.5,
    )

    assert list(motion)[1] == 0.1


# ---------------------------------------------------------------------------
# Issue #12: Count property and documented semantics
# ---------------------------------------------------------------------------


def test_count_property_matches_iteration():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(5),
    )

    assert motion.count == 3

    assert len(list(motion)) == motion.count


def test_count_property_exact_division():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(20),
        step=math.radians(5),
    )

    assert motion.count == 5


def test_count_property_partial_step():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=1.0,
        step=0.3,
    )

    # 0.0, 0.3, 0.6, 0.9 → 4 angles
    assert motion.count == 4


def test_max_angle_zero_yields_single_point():
    """
    Issue #12: max_angle=0.0 yields exactly one point
    (start_angle).  This is intentional and used by
    tests that need a single-point simulation.
    """

    motion = MotionRange(
        start_angle=math.radians(42),
        max_angle=0.0,
        step=math.radians(1),
    )

    angles = list(motion)

    assert len(angles) == 1

    assert math.isclose(angles[0], math.radians(42))

    assert motion.count == 1


def test_angles_are_start_plus_multiples():
    """
    Issue #12: Each angle is start_angle + i * step,
    not accumulated, to avoid FP drift.
    """

    motion = MotionRange(
        start_angle=1.0,
        max_angle=3.0,
        step=1.0,
    )

    angles = list(motion)

    assert math.isclose(angles[0], 1.0)
    assert math.isclose(angles[1], 2.0)
    assert math.isclose(angles[2], 3.0)


def test_large_iteration_count_is_stable():
    """
    Issue #12: With the old accumulation approach
    (travelled += step), 10000 iterations of
    radians(0.1) would drift.  The integer-based
    count and i*step approach must remain exact.
    """

    motion = MotionRange(
        start_angle=0.0,
        max_angle=999 * math.radians(0.1),
        step=math.radians(0.1),
    )

    angles = list(motion)

    assert len(angles) == 1000

    # The last angle should be 999 * radians(0.1).
    assert math.isclose(
        angles[-1],
        999 * math.radians(0.1),
        rel_tol=1e-12,
    )


def test_count_with_negative_direction():

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(10),
        step=math.radians(2),
        direction=-1,
    )

    assert motion.count == 6

    angles = list(motion)

    assert len(angles) == 6

    assert math.isclose(angles[0], 0.0)
    assert math.isclose(angles[-1], math.radians(-10))


def test_iter_is_reusable():
    """
    MotionRange is frozen and __iter__ creates a new
    iterator each time, so it can be iterated multiple
    times.
    """

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(5),
        step=math.radians(1),
    )

    first = list(motion)
    second = list(motion)

    assert first == second
    assert len(first) == 6