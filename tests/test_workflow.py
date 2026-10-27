"""tests/test_workflow.py

Unit tests for the interactive optimization
workflow in gui/workflow.py.
"""

from __future__ import annotations

import math

import pytest

from analysis.target_curve import (
    DiscreteTargetCurve,
)

from gui.workflow import (
    OptimizationSettings,
    WorkflowError,
    apply_parameter_overrides,
    load_inputs,
    mechanism_csv_text,
    optimized_values,
    parameter_template,
    preview,
    run_optimization,
    with_mechanism_levers,
    with_target_points,
)

MECHANISM_CSV = """\
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,50,150,50,30,150,90,0,0,0,0,0,1,,2
2,50,150,60,30,150,90,100,0,0,0,0,1,1,
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


def test_load_inputs_defaults_weights_to_one(inputs):
    assert isinstance(
        inputs.target_curve,
        DiscreteTargetCurve,
    )
    assert inputs.target_curve.weights == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )


def test_load_inputs_reads_target_weights():
    weighted_target = (
        "input_angle,output_angle,weight\n"
        "60,90,100\n"
        "70,100,1\n"
        "80,110,1\n"
        "90,120,100\n"
        "100,130,1\n"
        "110,140,1\n"
        "120,150,100\n"
    )
    loaded = load_inputs(
        MECHANISM_CSV,
        weighted_target,
    )
    assert loaded.target_curve.weights == (
        100.0,
        1.0,
        1.0,
        100.0,
        1.0,
        1.0,
        100.0,
    )


def test_load_inputs_reads_partial_weights():
    weighted_target = (
        "input_angle,output_angle,weight\n"
        "60,90,\n"
        "70,100,5\n"
        "80,110,\n"
    )
    loaded = load_inputs(
        MECHANISM_CSV,
        weighted_target,
    )
    assert loaded.target_curve.weights == (
        1.0,
        5.0,
        1.0,
    )


def test_with_target_points_replaces_curve(inputs):
    edited = with_target_points(
        inputs,
        [60.0, 75.0, 90.0],
        [95.0, 105.0, 125.0],
        [2.0, 0.0, 5.0],
    )
    assert edited.point_count == 3
    assert edited.input_angles[0] == (
        math.radians(60.0)
    )
    assert edited.output_angles[2] == (
        math.radians(125.0)
    )
    assert edited.target_curve.weights == (
        2.0,
        0.0,
        5.0,
    )
    assert edited.definition is inputs.definition


def test_with_target_points_rejects_unsorted_angles(inputs):
    with pytest.raises(WorkflowError):
        with_target_points(
            inputs,
            [60.0, 50.0],
            [90.0, 80.0],
            [1.0, 1.0],
        )


def test_with_target_points_rejects_negative_weight(inputs):
    with pytest.raises(WorkflowError):
        with_target_points(
            inputs,
            [60.0, 70.0],
            [90.0, 100.0],
            [1.0, -1.0],
        )


def test_with_target_points_clamps_large_weight(inputs):
    edited = with_target_points(
        inputs,
        [60.0, 70.0],
        [90.0, 100.0],
        [1.0, 1000.0],
    )
    assert edited.target_curve.weights == (
        1.0,
        100.0,
    )


def test_with_mechanism_levers_replaces_definition(inputs):
    edited = with_mechanism_levers(
        inputs,
        [
            {
                "id": 1,
                "length_min": 60.0,
                "length_max": 160.0,
                "length_start": 70.0,
                "angle_min": 50.0,
                "angle_max": 130.0,
                "angle_start": 80.0,
                "pivot_x": 0.0,
                "pivot_y": 0.0,
                "pivot_z": 0.0,
                "axis_x": 0.0,
                "axis_y": 0.0,
                "axis_z": 1.0,
            },
            {
                "id": 2,
                "length_min": 50.0,
                "length_max": 150.0,
                "length_start": 60.0,
                "angle_min": 60.0,
                "angle_max": 120.0,
                "angle_start": 90.0,
                "pivot_x": 100.0,
                "pivot_y": 0.0,
                "pivot_z": 0.0,
                "axis_x": 0.0,
                "axis_y": 0.0,
                "axis_z": 1.0,
                "coupled": 1,
            },
        ],
    )
    assert len(edited.definition.levers) == 2
    lever1 = edited.definition.levers[0]
    assert lever1.length_start == 70.0
    assert lever1.angle_min == math.radians(50.0)
    assert edited.target_curve is inputs.target_curve


def test_with_mechanism_levers_roundtrip_via_csv(inputs):
    rows = [
        {
            "id": lever.id,
            "length_min": lever.length_min,
            "length_max": lever.length_max,
            "length_start": lever.length_start,
            "angle_min": math.degrees(lever.angle_min),
            "angle_max": math.degrees(lever.angle_max),
            "angle_start": math.degrees(lever.angle_start),
            "pivot_x": lever.pivot.x,
            "pivot_y": lever.pivot.y,
            "pivot_z": lever.pivot.z,
            "axis_x": lever.axis.x,
            "axis_y": lever.axis.y,
            "axis_z": lever.axis.z,
            "ref_x": (
                lever.reference_direction.x
                if lever.reference_direction
                else None
            ),
            "ref_y": (
                lever.reference_direction.y
                if lever.reference_direction
                else None
            ),
            "ref_z": (
                lever.reference_direction.z
                if lever.reference_direction
                else None
            ),
            "driver": lever.driver,
            "coupled": lever.coupled,
            "pivot_on": lever.pivot_on,
        }
        for lever in inputs.definition.levers
    ]
    edited = with_mechanism_levers(inputs, rows)
    assert edited.definition.levers == (
        inputs.definition.levers
    )


def test_with_mechanism_levers_rejects_duplicate_ids(inputs):
    row = {
        "id": 1,
        "length_min": 50.0,
        "length_max": 150.0,
        "length_start": 50.0,
        "angle_min": 60.0,
        "angle_max": 120.0,
        "angle_start": 90.0,
        "pivot_x": 0.0,
        "pivot_y": 0.0,
        "pivot_z": 0.0,
        "axis_x": 0.0,
        "axis_y": 0.0,
        "axis_z": 1.0,
    }
    with pytest.raises(WorkflowError):
        with_mechanism_levers(inputs, [row, dict(row)])


def test_with_mechanism_levers_rejects_missing_value(inputs):
    row = {
        "id": 1,
        "length_min": 50.0,
    }
    with pytest.raises(WorkflowError):
        with_mechanism_levers(inputs, [row])


def test_with_mechanism_levers_rejects_invalid_pivot_on(inputs):
    row = {
        "id": 1,
        "length_min": 50.0,
        "length_max": 150.0,
        "length_start": 50.0,
        "angle_min": 60.0,
        "angle_max": 120.0,
        "angle_start": 90.0,
        "pivot_x": 0.0,
        "pivot_y": 0.0,
        "pivot_z": 0.0,
        "axis_x": 0.0,
        "axis_y": 0.0,
        "axis_z": 1.0,
        "pivot_on": "not-a-reference",
    }
    with pytest.raises(WorkflowError):
        with_mechanism_levers(inputs, [row])


def test_with_mechanism_levers_rejects_empty_definition(inputs):
    with pytest.raises(WorkflowError):
        with_mechanism_levers(inputs, [])


def test_with_mechanism_levers_preserves_coupled_priority(inputs):
    row = {
        "id": 1,
        "length_min": 50.0,
        "length_max": 150.0,
        "length_start": 50.0,
        "angle_min": 60.0,
        "angle_max": 120.0,
        "angle_start": 90.0,
        "pivot_x": 0.0,
        "pivot_y": 0.0,
        "pivot_z": 0.0,
        "axis_x": 0.0,
        "axis_y": 0.0,
        "axis_z": 1.0,
        "driver": 2,
        "coupled": 3,
    }
    edited = with_mechanism_levers(
        inputs,
        [row],
    )
    lever = edited.definition.levers[0]
    assert lever.coupled == 3
    assert lever.driver is None


def test_optimized_values_maps_definition(inputs):
    values = optimized_values(
        inputs.definition,
    )
    assert values == {
        "lever.1.length": 50.0,
        "lever.1.angle": math.radians(90.0),
        "lever.1.pivot.x": 0.0,
        "lever.1.pivot.y": 0.0,
        "lever.2.length": 60.0,
        "lever.2.angle": math.radians(90.0),
        "lever.2.pivot.x": 100.0,
        "lever.2.pivot.y": 0.0,
    }


def test_optimized_values_matches_template_names(inputs):
    template = parameter_template(inputs)
    values = optimized_values(
        inputs.definition,
    )
    assert set(values) == set(
        parameter.name
        for parameter in template.parameters
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


def test_preview_honours_overridden_template(inputs):
    template = parameter_template(inputs)
    overrides = {
        "lever.1.length": (
            50.0,
            150.0,
            60.0,
        ),
    }
    overridden = apply_parameter_overrides(
        template,
        overrides,
    )
    html = preview(
        inputs,
        overridden,
    )
    assert "<html" in html
    assert (
        math.degrees(
            overridden.get(
                "lever.1.length",
            ).value,
        )
        != math.degrees(
            template.get(
                "lever.1.length",
            ).value,
        )
    )


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


def test_run_optimization_parallel_matches_sequential(inputs):
    """Parallel evaluation (workers > 1) must produce the same
    best score as sequential evaluation for the same seed
    (issue #62)."""
    common = dict(
        population_size=20,
        children_count=20,
        selection_count=6,
        max_generations=5,
        seed=7,
    )

    sequential = run_optimization(
        inputs,
        parameter_template(inputs),
        OptimizationSettings(
            workers=1,
            **common,
        ),
    )

    parallel = run_optimization(
        inputs,
        parameter_template(inputs),
        OptimizationSettings(
            workers=2,
            **common,
        ),
    )

    assert math.isclose(
        sequential.best_score,
        parallel.best_score,
        rel_tol=1e-12,
    )


def test_run_optimization_gradient_weight_setting(inputs):
    """gradient_weight=0 must disable the gradient term and
    still produce a valid result (issue #65)."""
    result = run_optimization(
        inputs,
        parameter_template(inputs),
        OptimizationSettings(
            population_size=20,
            children_count=20,
            selection_count=6,
            max_generations=5,
            seed=7,
            workers=1,
            gradient_weight=0.0,
        ),
    )
    assert math.isfinite(result.best_score)


def test_middle_position_index_rounds_half_up():
    from gui.workflow import (
        middle_position_index,
    )
    assert middle_position_index(7) == 4
    assert middle_position_index(6) == 3
    assert middle_position_index(2) == 1
    assert middle_position_index(1) == 1


def test_anchored_input_angles_reference_middle(
    inputs,
):
    from gui.workflow import (
        anchored_input_angles,
        middle_input_angle,
    )
    middle = middle_input_angle(inputs)
    drive_start = math.radians(90.0)
    angles = anchored_input_angles(
        inputs,
        drive_start,
    )
    index = len(angles) // 2
    assert angles[index] == pytest.approx(
        drive_start,
    )
    offsets = [
        angle - drive_start
        for angle in angles
    ]
    expected = [
        angle - middle
        for angle in inputs.input_angles
    ]
    assert offsets == pytest.approx(expected)
    # The angular distances between the support
    # points are preserved.
    deltas = [
        angles[i + 1] - angles[i]
        for i in range(len(angles) - 1)
    ]
    original = [
        inputs.input_angles[i + 1]
        - inputs.input_angles[i]
        for i in range(
            len(inputs.input_angles) - 1
        )
    ]
    assert deltas == pytest.approx(original)


def test_anchored_output_angles_reference_middle(
    inputs,
):
    from gui.workflow import (
        anchored_output_angles,
        middle_output_angle,
    )
    middle = middle_output_angle(inputs)
    output_start = math.radians(90.0)
    angles = anchored_output_angles(
        inputs,
        output_start,
    )
    index = len(angles) // 2
    assert angles[index] == pytest.approx(
        output_start,
    )
    expected = [
        angle - middle + output_start
        for angle in inputs.output_angles
    ]
    assert angles == pytest.approx(expected)


def test_preview_curves_in_target_system(
    inputs,
):
    """Seven support points: in the preview the Soll
    curve is drawn exactly as specified by the target
    curve (in the system of the Vorgabewinkel), and
    the Ist curve is shifted into the same system.
    Issue #78."""
    import json
    import re

    template = parameter_template(inputs)
    html = preview(inputs, template)
    match = re.search(
        r'<script id="plot-data" '
        r'type="application/json">'
        r"(.*?)</script>",
        html,
        re.S,
    )
    data = json.loads(match.group(1))
    curves = {
        series["label"]: series
        for series in data["curves"]
    }
    target = curves["Soll-Ausgangskurve"]
    assert target["input_angles"] == (
        pytest.approx(
            [
                round(
                    math.degrees(angle),
                    1,
                )
                for angle in inputs.input_angles
            ]
        )
    )
    assert target["output_angles"] == (
        pytest.approx(
            [
                round(
                    math.degrees(angle),
                    1,
                )
                for angle in inputs.output_angles
            ]
        )
    )


def test_preview_drive_angle_at_middle_position(
    inputs,
):
    """Seven support points: the drive lever must sit
    at its start angle at the fourth position."""
    template = parameter_template(inputs)
    html = preview(inputs, template)
    assert "<html" in html
