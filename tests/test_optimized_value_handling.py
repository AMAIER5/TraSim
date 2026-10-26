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
            "length_start": 82.0,
            "length_optimized": 90.0,
            "length_adopt": False,
            "angle_start": 10.0,
            "angle_optimized": 12.0,
            "angle_adopt": False,
        },
        {
            "id": 2,
            "length_start": 50.0,
            "length_optimized": None,
            "length_adopt": False,
            "angle_start": -5.0,
            "angle_optimized": None,
            "angle_adopt": False,
        },
    ]


def _snapshot() -> dict:
    return {
        1: {"length": 82.0, "angle": 10.0},
        2: {"length": 50.0, "angle": -5.0},
    }


def _optimized_map() -> dict:
    return {
        "lever.1.length": 90.0,
        "lever.1.angle": math.radians(12.0),
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
    assert values[2] == (None, None)


def test_display_is_cleared_after_manual_start_change():
    rows = _rows()
    rows[0]["length_start"] = 60.0
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


def test_display_is_empty_without_result():
    values = optimized_display_values(
        _rows(),
        {},
        {},
    )
    assert all(
        value == (None, None)
        for value in values.values()
    )


def test_adopt_flag_overwrites_start_value():
    rows = _rows()
    rows[0]["length_adopt"] = True
    adopted = (
        adopt_single_optimized_values(rows)
    )
    assert adopted is True
    assert rows[0]["length_start"] == 90.0
    assert rows[0]["length_adopt"] is False


def test_adopt_flag_without_optimized_value_is_ignored():
    rows = _rows()
    rows[1]["length_adopt"] = True
    adopted = (
        adopt_single_optimized_values(rows)
    )
    assert adopted is False
    assert rows[1]["length_start"] == 50.0


def test_adopted_value_is_cleared_after_further_change():
    rows = _rows()
    rows[0]["length_adopt"] = True
    adopt_single_optimized_values(rows)
    assert rows[0]["length_start"] == 90.0
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
    rows[0]["length_start"] = 70.0
    values = optimized_display_values(
        rows,
        _optimized_map(),
        snapshot,
    )
    assert values[1][0] is None
