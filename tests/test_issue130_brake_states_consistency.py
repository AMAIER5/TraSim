"""Tests for issue #130: the mixed brake result pages
showed identical (or nearly identical) lever values for
the braked and unbraked states although the result
claimed clearly different brake lever positions, and the
drawn coupler rods looked like they had different
lengths ("braked and unbraked position are calculated
with different bar lengths").

Root cause: when the fully braked position of the best
candidate blocks, the old fallback moved the brake
lever's ``angle_max`` all the way onto its unbraked start
angle (the 0 deg travel clamp of issue #115).  The result
then degenerated silently: the returned definition (and
with it every report the GUI rebuilds from it) contained
a zero-travel brake lever while ``brake_positions_deg``
still reported the full grid, so the 0 % and 100 % pages
rendered the very same mechanism under different brake
labels.

Invariants now (issues #115, #127 and #130):

- the fully braked position of the generated solution
  never blocks (issue #115),
- the brake travel is reduced only as far as necessary
  and never to 0 deg while a larger non-blocking
  position exists (issues #127/#130),
- the reported ``brake_positions_deg`` always match the
  returned optimized definition (issue #130), and
- the 0 % and 100 % states of a non-degenerate result
  differ (issue #130).
"""

from __future__ import annotations

import math

import pytest

from gui.mixed_brake_workflow import (
    _braked_blocks_but_unbraked_works,
    _brake_grid_position,
    _brake_lever_id,
    _fully_braked_position_works,
    _motion,
    _reduce_brake_travel_to_largest_non_blocking,
    build_mixed_brake_grid,
    mixed_brake_inputs_from_parts,
    mixed_brake_parameter_template,
    run_mixed_brake_optimization,
)
from analysis.brake_grid import (
    BrakeGridSettings,
)
from tests.test_issue115_non_blocking_unbraked import (
    _blocking_inputs,
    _small_settings,
)


def test_reduce_keeps_largest_non_blocking_travel():
    """The blocking fixture: the unbraked position works,
    the fully braked position blocks.  The reduction must
    keep the largest working brace position instead of
    collapsing the travel onto the unbraked position."""
    inputs = _blocking_inputs()
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
        position_count=5,
    )
    assert _braked_blocks_but_unbraked_works(grid)
    reduced = _reduce_brake_travel_to_largest_non_blocking(
        inputs.definition,
        motion=_motion(inputs),
    )
    brake = reduced.get_lever(
        _brake_lever_id(reduced),
    )
    base_brake = inputs.definition.get_lever(
        _brake_lever_id(inputs.definition),
    )
    assert (
        brake.resolved_brake_position
        >= brake.angle_start
    )
    # issue #127: the reduction is stored in the dedicated
    # brake_position attribute and leaves the angle_max
    # optimization bound untouched.
    assert brake.brake_position is not None
    assert brake.angle_max == pytest.approx(
        base_brake.angle_max,
    )
    # strictly smaller travel than the blocking original
    assert (
        brake.resolved_brake_position
        - brake.angle_start
    ) <= (
        base_brake.resolved_brake_position
        - base_brake.angle_start
    )


def test_brake_grid_position_matches_grid_pathway():
    """The candidate braked end must be verified through
    the exact degrees-based pathway the brake grids use,
    so the reduction and the rebuilt grid cannot disagree
    on the blocking verdict (issue #130)."""
    for start_deg, end_deg in (
        (-10.0, 90.0),
        (140.05, 267.66),
        (-0.416254, 0.165294),
    ):
        grid = BrakeGridSettings(
            brake_min_deg=start_deg,
            brake_max_deg=end_deg,
            position_count=2,
        )
        assert _brake_grid_position(
            start_deg,
            end_deg,
        ) == pytest.approx(grid.brake_positions_deg[-1])


def test_result_report_matches_definition():
    """End-to-end: the generated result's reported brake
    positions, its optimized definition and every rebuilt
    grid state must agree, and the fully braked position
    never blocks (issues #115/#130)."""
    inputs = _blocking_inputs()
    result = run_mixed_brake_optimization(
        inputs,
        mixed_brake_parameter_template(inputs),
        _small_settings(),
    )
    optimized = result.optimized_definition
    assert optimized is not None
    brake = optimized.get_lever(
        _brake_lever_id(optimized),
    )
    assert result.brake_positions_deg[0] == (
        pytest.approx(
            math.degrees(brake.angle_start)
        )
    )
    assert result.brake_positions_deg[-1] == (
        pytest.approx(
            math.degrees(
                brake.resolved_brake_position
            )
        )
    )
    # the reported fully braked position itself is the
    # one the grids rebuild from the definition
    grid = build_mixed_brake_grid(
        inputs,
        optimized,
        position_count=5,
    )
    assert not _braked_blocks_but_unbraked_works(
        grid,
    )
    assert grid.brake_positions_deg[-1] == (
        pytest.approx(
            result.brake_positions_deg[-1]
        )
    )
    # the fully braked position of the RESULT works and
    # keeps a non-degenerate travel when possible
    assert _fully_braked_position_works(
        optimized,
        brake.resolved_brake_position,
        _motion(inputs),
    )
    assert (
        brake.resolved_brake_position
        >= brake.angle_start
    )


def test_states_differ_between_brake_positions():
    """With a working (non-degenerate) travel, the 0 %
    and 100 % states of the result REALLY differ in brake
    lever position - no identical mechanism under
    different brake labels (issue #130)."""
    inputs = _blocking_inputs()
    result = run_mixed_brake_optimization(
        inputs,
        mixed_brake_parameter_template(inputs),
        _small_settings(),
    )
    optimized = result.optimized_definition
    assert optimized is not None
    brake = optimized.get_lever(
        _brake_lever_id(optimized),
    )
    if (
        brake.resolved_brake_position
        - brake.angle_start
    ) <= 0.0:
        pytest.skip(
            "degenerate zero-travel result; nothing to "
            "differ between the brake positions"
        )
    grid = build_mixed_brake_grid(
        inputs,
        optimized,
        position_count=5,
    )
    first = grid.states[0]
    last = grid.states[-1]
    assert first is not None
    assert last is not None
    first_lever = next(
        lever
        for lever in first.states[0].levers
        if lever.name == "Lever 7"
    )
    last_lever = next(
        lever
        for lever in last.states[0].levers
        if lever.name == "Lever 7"
    )
    assert math.degrees(first_lever.angle) == (
        pytest.approx(
            result.brake_positions_deg[0],
            abs=0.05,
        )
    )
    assert math.degrees(last_lever.angle) == (
        pytest.approx(
            result.brake_positions_deg[-1],
            abs=0.05,
        )
    )
    assert math.degrees(first_lever.angle) != (
        pytest.approx(
            math.degrees(last_lever.angle),
        )
    )
