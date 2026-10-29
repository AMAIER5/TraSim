"""tests/test_optimized_value_handling.py

Tests for the single-table optimized-value
handling introduced with the unified parameter
table:

- ``optimized_display_values``: an optimized
  value is displayed only while the corresponding
  start value still matches the snapshot of the
  optimization run (staleness = cleared cell).
- ``adopt_single_optimized_values``: a set
  adopt flag adopts the displayed optimized
  value as the new start value and clears the
  flag.
"""

from __future__ import annotations

import math

from gui.workflow import (
    adopt_single_optimized_values,
    optimized_display_values,
)


def _rows() -> list[dict]:
    return [
        {
            "id": 1,
            "length": 82.0,
            "length_optimized": 90.0,
            "length_adopt": False,
            "angle": 10.0,
            "angle_optimized": 12.0,
            "angle_adopt": False,
            "pivot_x": 100.0,
            "pivot_x_optimized": 110.0,
            "pivot_x_adopt": False,
            "pivot_y": -40.0,
            "pivot_y_optimized": -35.0,
            "pivot_y_adopt": False,
            "pivot_z": 5.0,
            "pivot_z_optimized": 7.0,
            "pivot_z_adopt": False,
        },
        {
            "id": 2,
            "length": 50.0,
            "length_optimized": None,
            "length_adopt": False,
            "angle": -5.0,
            "angle_optimized": None,
            "angle_adopt": False,
            "pivot_x": 0.0,
            "pivot_x_optimized": None,
            "pivot_x_adopt": False,
            "pivot_y": 0.0,
            "pivot_y_optimized": None,
            "pivot_y_adopt": False,
            "pivot_z": 0.0,
            "pivot_z_optimized": None,
            "pivot_z_adopt": False,
        },
    ]


def _snapshot() -> dict:
    return {
        1: {
            "length": 82.0,
            "angle": 10.0,
            "pivot_x": 100.0,
            "pivot_y": -40.0,
            "pivot_z": 5.0,
        },
        2: {
            "length": 50.0,
            "angle": -5.0,
            "pivot_x": 0.0,
            "pivot_y": 0.0,
            "pivot_z": 0.0,
        },
    }


def _optimized_map() -> dict:
    return {
        "lever.1.length": 90.0,
        "lever.1.angle": math.radians(12.0),
        "lever.1.pivot.x": 110.0,
        "lever.1.pivot.y": -35.0,
        "lever.1.pivot.z": 7.0,
    }


def test_display_shows_optimized_values_for_matching_starts():
    values = optimized_display_values(
        _rows(),
        _optimized_map(),
        _snapshot(),
    )
    assert values[1][0] == 90.0
    assert math.isclose(
        values[1][1],
        12.0,
    )
    assert values[1][2] == 110.0
    assert values[1][3] == -35.0
    assert values[1][4] == 7.0
    assert values[2] == (
        None,
        None,
        None,
        None,
        None,
    )


def test_display_is_cleared_after_manual_start_change():
    rows = _rows()
    rows[0]["length"] = 60.0
    values = optimized_display_values(
        rows,
        _optimized_map(),
        _snapshot(),
    )
    assert values[1][0] is None
    assert math.isclose(
        values[1][1],
        12.0,
    )


def test_display_is_cleared_after_manual_pivot_change():
    rows = _rows()
    rows[0]["pivot_x"] = 90.0
    values = optimized_display_values(
        rows,
        _optimized_map(),
        _snapshot(),
    )
    assert values[1][2] is None
    assert values[1][3] == -35.0


def test_display_is_empty_without_result():
    values = optimized_display_values(
        _rows(),
        {},
        {},
    )
    assert all(
        value == (
            None,
            None,
            None,
            None,
            None,
        )
        for value in values.values()
    )


def test_adopt_flag_overwrites_start_value():
    rows = _rows()
    rows[0]["length_adopt"] = True
    adopted = (
        adopt_single_optimized_values(rows)
    )
    assert adopted is True
    assert rows[0]["length"] == 90.0
    assert rows[0]["length_adopt"] is False


def test_adopt_flag_without_optimized_value_is_ignored():
    rows = _rows()
    rows[1]["length_adopt"] = True
    adopted = (
        adopt_single_optimized_values(rows)
    )
    assert adopted is False
    assert rows[1]["length"] == 50.0


def test_adopted_value_is_cleared_after_further_change():
    rows = _rows()
    rows[0]["length_adopt"] = True
    adopt_single_optimized_values(rows)
    assert rows[0]["length"] == 90.0
    snapshot = {
        1: {"length": 90.0, "angle": 10.0}
    }
    values = optimized_display_values(
        rows,
        _optimized_map(),
        snapshot,
    )
    assert values[1][0] == 90.0
    assert math.isclose(
        values[1][1],
        12.0,
    )
    rows[0]["length"] = 70.0
    values = optimized_display_values(
        rows,
        _optimized_map(),
        snapshot,
    )
    assert values[1][0] is None


def test_adopt_flag_overwrites_pivot_start_value():
    rows = _rows()
    rows[0]["pivot_x_adopt"] = True
    rows[0]["pivot_y_adopt"] = True
    adopted = (
        adopt_single_optimized_values(rows)
    )
    assert adopted is True
    assert rows[0]["pivot_x"] == 110.0
    assert rows[0]["pivot_y"] == -35.0
    assert rows[0]["pivot_x_adopt"] is False
    assert rows[0]["pivot_y_adopt"] is False


def test_adopt_flag_without_pivot_optimized_value_is_ignored():
    rows = _rows()
    rows[1]["pivot_x_adopt"] = True
    adopted = (
        adopt_single_optimized_values(rows)
    )
    assert adopted is False
    assert rows[1]["pivot_x"] == 0.0
