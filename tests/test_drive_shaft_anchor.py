"""tests/test_drive_shaft_anchor.py

Tests for the virtual drive shaft semantics: the target
curve's input angle describes the rotation of a virtual
drive shaft; the workflows form the initial angular offset
``drive lever angle - MEAN of all prescribed input values``
so the start lever's displayed initial angle is that mean
(e.g. 0 deg).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gui.workflow import (
    anchored_input_angles,
    drive_shaft_offset,
    load_inputs,
    mean_input_angle,
    with_target_points,
)

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)


def _loaded_inputs():
    return load_inputs(
        (
            EXAMPLES_DIR / "pivot_shift_mechanism.csv"
        ).read_text(encoding="utf-8"),
        (
            EXAMPLES_DIR
            / "target_curve_unbraked.csv"
        ).read_text(encoding="utf-8"),
    )


def _asymmetric_inputs():
    """Asymmetric support: the MEAN of the prescribed input
    values (22/6 deg) differs from the middle index support
    point (0 deg), so both anchor conventions are
    distinguishable."""
    inputs = _loaded_inputs()
    return with_target_points(
        inputs,
        [-20.0, -12.0, 0.0, 10.0, 20.0, 24.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )


def test_mean_input_angle_is_arithmetic_mean():
    inputs = _asymmetric_inputs()
    angles = inputs.input_angles
    assert mean_input_angle(inputs) == sum(
        angles,
    ) / len(angles)
    assert math.degrees(
        mean_input_angle(inputs),
    ) == pytest.approx(22.0 / 6.0)


def test_anchor_uses_mean_of_all_values():
    inputs = _asymmetric_inputs()
    mean = mean_input_angle(inputs)
    drive_angle = math.radians(90.0)
    anchored = anchored_input_angles(
        inputs,
        drive_angle=drive_angle,
    )
    expected = tuple(
        angle - mean + drive_angle
        for angle in inputs.input_angles
    )
    assert anchored == expected
    # The middle INDEX support (0 deg) is NOT the anchor:
    assert math.degrees(mean) != 0.0


def test_drive_shaft_offset_matches_start_minus_mean():
    inputs = _asymmetric_inputs()
    drive = inputs.definition.get_lever(1)
    offset = drive_shaft_offset(inputs)
    assert offset == (
        drive.angle_start
        - mean_input_angle(inputs)
    )


def test_symmetric_example_has_zero_offset():
    """The example mechanism's start lever sits at 0 deg with
    a symmetric support -10..10 deg, so the initial drive
    shaft offset of the loaded example is zero."""
    inputs = _loaded_inputs()
    assert drive_shaft_offset(inputs) == 0.0
