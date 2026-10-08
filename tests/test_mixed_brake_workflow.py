"""tests/test_mixed_brake_workflow.py

Tests for the GUI mixed brake (mixer) workflow:
- loading the example inputs validates the pivot_on topology
- the parameter template matches the mechanism definition
- a short optimization run over the 5-position brake grid
  reaches a finite best score and reports the minimum
  transmission angle
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gui.mixed_brake_workflow import (
    MixedBrakeSettings,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
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
        MECHANISM_FILE,
        UNBRAKED_FILE,
        BRAKED_FILE,
    )


def test_load_inputs_validates_topology(inputs):
    mixers = [
        lever
        for lever in inputs.definition.levers
        if lever.pivot_on is not None
    ]
    assert len(mixers) == 1
    assert inputs.brake_min_deg == (
        pytest.approx(180.0)
    )
    assert inputs.brake_max_deg == (
        pytest.approx(190.0)
    )


def test_load_inputs_rejects_mechanism_without_pivot_on(
    tmp_path,
):
    mechanism = tmp_path / "mechanism.csv"
    mechanism.write_text(
        "id,length_min,length_max,"
        "length_start,angle_min,"
        "angle_max,angle_start,"
        "pivot_x,pivot_y,pivot_z,"
        "axis_x,axis_y,axis_z,"
        "driver,coupled,pivot_on\n"
        "1,40,100,60,-12,12,0,"
        "0,0,0,0,0,1,,,\n"
        "2,40,200,100,-120,120,0,"
        "100,60,0,0,0,1,1,,\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkflowError):
        load_mixed_brake_inputs(
            mechanism,
            UNBRAKED_FILE,
            BRAKED_FILE,
        )


def test_parameter_template_matches(inputs):
    template = mixed_brake_parameter_template(
        inputs,
    )
    names = {
        parameter.name
        for parameter in template.parameters
    }
    assert "lever.1.length" in names
    assert "lever.1.angle" in names
    assert "lever.3.length" in names


def test_short_optimization_over_grid(inputs):
    settings = MixedBrakeSettings(
        population_size=8,
        children_count=8,
        selection_count=4,
        max_generations=3,
        brake_positions=5,
        transmission_angle_deg=15.0,
    )
    template = mixed_brake_parameter_template(
        inputs,
    )
    result = run_mixed_brake_optimization(
        inputs,
        template,
        settings,
    )
    assert len(result.brake_positions_deg) == 5
    assert result.brake_positions_deg[0] == (
        pytest.approx(180.0)
    )
    assert result.brake_positions_deg[-1] == (
        pytest.approx(190.0)
    )
    assert result.best_score < float("inf")
    assert result.generations >= 1
    assert result.fitness_history
    assert (
        result.min_transmission_angle_deg
        > 0.0
    )
    assert "<html" in result.html.lower() or (
        "TraSim" in result.html
    )


def test_progress_callback_is_called(inputs):
    settings = MixedBrakeSettings(
        population_size=6,
        children_count=6,
        selection_count=3,
        max_generations=2,
        brake_positions=3,
    )
    template = mixed_brake_parameter_template(
        inputs,
    )
    calls = []

    def progress(generation, best_score):
        calls.append((generation, best_score))

    run_mixed_brake_optimization(
        inputs,
        template,
        settings,
        progress=progress,
    )
    assert calls
