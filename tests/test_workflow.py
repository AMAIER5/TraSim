"""tests/test_workflow.py

Unit tests for the interactive optimization
workflow in gui/workflow.py.
"""

from __future__ import annotations

import math

import pytest

from gui.workflow import (
    OptimizationSettings,
    WorkflowError,
    apply_parameter_overrides,
    load_inputs,
    mechanism_csv_text,
    parameter_template,
    preview,
    run_optimization,
)

MECHANISM_CSV = """\
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,50,150,50,60,120,90,0,0,0,0,0,1,,2
2,50,150,60,60,120,90,100,0,0,0,0,1,1,
"""

TARGET_CSV = """\
input_angle,output_angle
60,90
70,100
80,110
90,120
100,130
110,140
120,150
"""


@pytest.fixture
def inputs():
    return load_inputs(
        MECHANISM_CSV,
        TARGET_CSV,
    )


def test_load_inputs_parses_files(inputs):
    assert len(inputs.definition.levers) == 2
    assert inputs.point_count == 7
    assert inputs.input_angles[0] == (
        math.radians(60)
    )


def test_load_inputs_rejects_invalid_csv():
    with pytest.raises(WorkflowError):
        load_inputs(
            "not,a,valid,mechanism",
            TARGET_CSV,
        )


def test_load_inputs_rejects_single_point_target():
    with pytest.raises(WorkflowError):
        load_inputs(
            MECHANISM_CSV,
            "input_angle,output_angle\n60,90\n",
        )


def test_load_inputs_detects_german_convention():
    german_target = (
        "input_angle;output_angle\n"
        "60,0;90,0\n"
        "70,0;100,0\n"
        "80,0;110,0\n"
    )
    loaded = load_inputs(
        MECHANISM_CSV,
        german_target,
    )
    assert (
        loaded.target_convention.is_german
    )
    assert (
        loaded.output_convention.is_german
    )
    assert loaded.input_angles == (
        loaded.input_angles
    )


def test_parameter_template_matches_definition(
    inputs,
):
    template = parameter_template(inputs)
    names = {
        parameter.name
        for parameter in template.parameters
    }
    assert names == {
        "lever.1.length",
        "lever.1.angle",
        "lever.1.pivot.x",
        "lever.1.pivot.y",
        "lever.2.length",
        "lever.2.angle",
        "lever.2.pivot.x",
        "lever.2.pivot.y",
    }


def test_apply_overrides_converts_angle_units(
    inputs,
):
    template = parameter_template(inputs)
    overrides = {
        "lever.1.angle": (
            30.0,
            150.0,
            90.0,
        ),
    }
    overridden = apply_parameter_overrides(
        template,
        overrides,
    )
    angle = overridden.get(
        "lever.1.angle",
    )
    assert angle.minimum == (
        math.radians(30.0)
    )
    assert angle.maximum == (
        math.radians(150.0)
    )
    assert angle.value == math.radians(90.0)


def test_apply_overrides_rejects_value_outside_range(
    inputs,
):
    template = parameter_template(inputs)
    with pytest.raises(ValueError):
        apply_parameter_overrides(
            template,
            {"lever.1.length": (
                50.0,
                100.0,
                200.0,
            )},
        )


def test_mechanism_csv_text_roundtrip(
    inputs,
):
    text = mechanism_csv_text(
        inputs.definition,
        inputs.output_convention,
    )
    assert "length_min" in text
    reloaded = load_inputs(
        text,
        TARGET_CSV,
    )
    assert len(reloaded.definition.levers) == 2


def test_preview_produces_html(inputs):
    html = preview(inputs)
    assert "<html" in html
    assert "Soll-Ausgangskurve" in html


def test_run_optimization_returns_result(inputs):
    settings = OptimizationSettings(
        population_size=20,
        children_count=20,
        selection_count=6,
        max_generations=5,
        seed=7,
    )
    calls: list[int] = []

    def progress(
        generation: int,
        best_score: float,
    ) -> None:
        calls.append(generation)

    result = run_optimization(
        inputs,
        parameter_template(inputs),
        settings,
        progress=progress,
    )
    assert calls
    assert (
        result.stop_reason
        is not None
    )
    assert result.generations <= 5
    assert math.isfinite(result.best_score)
    assert "<html" in result.html
    assert (
        "length_min"
        in result.mechanism_csv
    )
    assert len(
        result.fitness_history
    ) == result.generations + 1
