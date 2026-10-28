"""tests/test_mixed_brake_gui_workflow.py

Tests for the integrated mixed brake (mixer) workflow used
by the GUI:

- the brake grid builds one simulation state per brake
  position and blends the target curves
- the mechanism snapshots include the adjusting lever
  (Verstellhebel) the mixer lever pivots on
- one brake position can be simulated individually
  (0/25/50/75/100 % brake travel)
- the preview report shows all brake positions
- the optimization result carries the optimized mechanism
  definition and its CSV export
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gui.mixed_brake_workflow import (
    MixedBrakeSettings,
    build_mixed_brake_grid,
    build_mixed_brake_state,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
    mixed_brake_preview,
    run_mixed_brake_optimization,
)
from gui.workflow import WorkflowError

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)
MECHANISM_FILE = (
    EXAMPLES_DIR / "mixed_brake_mechanism.csv"
)
UNBRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_unbraked.csv"
)
BRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_braked.csv"
)


@pytest.fixture
def inputs():
    return load_mixed_brake_inputs(
        MECHANISM_FILE.read_text(encoding="utf-8"),
        UNBRAKED_FILE.read_text(encoding="utf-8"),
        BRAKED_FILE.read_text(encoding="utf-8"),
    )


def _plot_data(html: str) -> dict:
    match = re.search(
        r'<script id="plot-data" '
        r'type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_grid_builds_all_brake_positions(inputs):
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
    )
    assert grid.fractions == (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    assert grid.brake_positions_deg[0] == (
        pytest.approx(180.0)
    )
    assert grid.brake_positions_deg[-1] == (
        pytest.approx(190.0)
    )
    assert len(grid.states) == 5
    assert all(
        state is not None
        for state in grid.states
    )
    assert grid.min_transmission_angle_deg > 0.0


def test_snapshots_include_adjusting_lever(inputs):
    """Requirement 1.3: the snapshots also contain the
    lever the mixer pivot sits on (the adjusting lever,
    here lever 3, which drives no stage itself)."""
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
    )
    for state in grid.states:
        assert state is not None
        for snapshot in state.states:
            names = [
                lever.name
                for lever in snapshot.levers
            ]
            assert "Lever 3" in names


def test_build_state_per_brake_position(inputs):
    template = mixed_brake_parameter_template(
        inputs
    )
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    angles = []
    for fraction in fractions:
        state = build_mixed_brake_state(
            inputs,
            template,
            brake_fraction=fraction,
        )
        assert state.brake_fraction == fraction
        angles.append(state.brake_angle_deg)
    assert angles == sorted(angles)
    assert angles[0] == pytest.approx(180.0)
    assert angles[-1] == pytest.approx(190.0)


def test_build_state_rejects_invalid_fraction(inputs):
    template = mixed_brake_parameter_template(
        inputs
    )
    with pytest.raises(WorkflowError):
        build_mixed_brake_state(
            inputs,
            template,
            brake_fraction=1.5,
        )


def test_preview_report_shows_all_positions(inputs):
    html = mixed_brake_preview(inputs)
    payload = _plot_data(html)
    labels = [
        series["label"]
        for series in payload["curves"]
    ]
    assert "Soll Bremse 0 %" in labels
    assert "Soll Bremse 100 %" in labels
    assert "Bremse 0 %" in labels
    assert "Bremse 100 %" in labels
    assert payload["mechanism_states"]
    lever_names = [
        lever["name"]
        for lever in payload[
            "mechanism_states"
        ][0]["levers"]
    ]
    assert "Lever 3" in lever_names


def test_optimization_result_carries_definition_and_csv(
    inputs,
):
    settings = MixedBrakeSettings(
        population_size=6,
        children_count=6,
        selection_count=3,
        max_generations=2,
        brake_positions=5,
    )
    template = mixed_brake_parameter_template(
        inputs
    )
    result = run_mixed_brake_optimization(
        inputs,
        template,
        settings,
    )
    assert result.optimized_definition is not None
    assert result.optimized_definition.levers
    assert "id," in result.mechanism_csv
    assert "pivot_on" in result.mechanism_csv


def test_mixed_inputs_from_parts_roundtrip(inputs):
    """The GUI rebuilds the mixed inputs from the
    (edited) standard definition and the unbraked
    target curve after every edit."""
    from gui.mixed_brake_workflow import (
        mixed_brake_inputs_from_parts,
    )

    rebuilt = mixed_brake_inputs_from_parts(
        inputs.definition,
        inputs.unbraked,
        inputs.braked,
    )
    assert rebuilt.brake_min_deg == (
        pytest.approx(inputs.brake_min_deg)
    )
    assert rebuilt.brake_max_deg == (
        pytest.approx(inputs.brake_max_deg)
    )
