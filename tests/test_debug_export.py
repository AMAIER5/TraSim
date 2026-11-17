"""Tests for the JSON debug export (download button beneath
the result HTML download)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from analysis.target_curve import TargetCurve
from gui.debug_export import (
    build_debug_dict,
    debug_export_json,
    debug_result_info,
    definition_dict,
    parameters_dict,
    target_curve_dict,
)
from mechanism_io.csv_reader import CsvReader
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)

MECHANISM_CSV = """
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled,pivot_on
1,40,100,60,-40,40,0,0,60,0,0,0,1,,,lever3
2,20,80,50,-60,60,0,50,10,0,0,0,1,1,,
3,40,200,100,-30,30,0,180,0,0,0,0,1,2,,
4,20,80,50,-60,60,0,0,10,0,0,0,1,3,,
""".strip()


def _definition(tmp_path):
    path = tmp_path / "mechanism.csv"
    path.write_text(MECHANISM_CSV, encoding="utf-8")
    return CsvReader.read_mechanism(path)


def test_definition_dict_uses_degrees(tmp_path):
    definition = _definition(tmp_path)
    data = definition_dict(definition)
    lever3 = next(
        lever
        for lever in data["levers"]
        if lever["id"] == 3
    )
    assert lever3["angle_start_deg"] == 0.0
    assert lever3["angle_max_deg"] == pytest.approx(30.0)
    assert lever3["pivot"] == {"x": 180.0, "y": 0.0, "z": 0.0}
    assert lever3["pivot_on"] == "lever3" or lever3["id"] != 1


def test_parameters_dict_marks_fixed_and_degrees(tmp_path):
    definition = _definition(tmp_path)
    template = CsvParameterFactory.create(definition)
    entries = {
        entry["name"]: entry
        for entry in parameters_dict(template)
    }
    angle = entries["lever.3.angle"]
    assert angle["value_deg"] == 0.0
    assert angle["minimum_deg"] == pytest.approx(-30.0)
    assert angle["maximum_deg"] == pytest.approx(30.0)
    assert angle["fixed"] is False
    length = entries["lever.3.length"]
    assert "value_deg" not in length
    assert length["fixed"] is False


def test_target_curve_dict_samples_support_points():
    curve = TargetCurve.from_points(
        input_angles=(
            math.radians(-10),
            math.radians(0),
            math.radians(10),
        ),
        output_angles=(
            math.radians(-20),
            math.radians(0),
            math.radians(20),
        ),
    )
    data = target_curve_dict(
        "target_curve",
        curve,
        (math.radians(-10), math.radians(0), math.radians(10)),
    )
    assert data["name"] == "target_curve"
    assert data["input_angles_deg"] == pytest.approx([-10.0, 0.0, 10.0])
    assert data["output_angles_deg"] == pytest.approx([-20.0, 0.0, 20.0])
    assert data["weights"] is None


def test_debug_export_json_contains_all_sections(tmp_path):
    definition = _definition(tmp_path)
    template = CsvParameterFactory.create(definition)
    curve = TargetCurve.from_points(
        input_angles=(math.radians(-10), math.radians(10)),
        output_angles=(math.radians(-20), math.radians(20)),
    )
    result = SimpleNamespace(
        stop_reason="max_generations_reached",
        best_score=0.0123,
        generations=17,
        fitness_history=(0.5, 0.2, 0.0123),
        best_values={"lever.3.angle": math.radians(5.0)},
        brake_positions_deg=(0.0, 15.0, 30.0),
        min_transmission_angle_deg=25.5,
    )
    payload = json.loads(
        debug_export_json(
            mode="mixed_brake",
            initial_definition=definition,
            target_curves=[
                target_curve_dict(
                    "unbraked",
                    curve,
                    (math.radians(-10), math.radians(10)),
                )
            ],
            optimization_settings={
                "population_size": 40,
                "seed": 42,
            },
            parameters=parameters_dict(template),
            result=debug_result_info(result),
            optimized_definition=definition,
            initial_mechanism_csv="id,...\n",
            optimized_mechanism_csv="id,...\n",
            extra={"brake_max_deg": 30.0},
        )
    )
    assert payload["mode"] == "mixed_brake"
    assert payload["initial_mechanism"]["levers"]
    assert payload["optimized_mechanism"]["levers"]
    assert payload["target_curves"][0]["name"] == "unbraked"
    assert payload["optimization_settings"]["seed"] == 42
    assert any(
        entry["name"] == "lever.3.angle"
        for entry in payload["optimizable_parameters"]
    )
    assert payload["result"]["stop_reason"] == (
        "max_generations_reached"
    )
    assert payload["result"]["fitness_history"] == [
        0.5,
        0.2,
        0.0123,
    ]
    assert payload["result"]["best_values"][
        "lever.3.angle_deg"
] == pytest.approx(5.0)
    assert payload["result"]["brake_positions_deg"] == (
        pytest.approx([0.0, 15.0, 30.0])
    )
    assert payload["result"]["min_transmission_angle_deg"] == (
        25.5
    )
    assert payload["initial_mechanism_csv"] == "id,...\n"
    assert payload["optimized_mechanism_csv"] == "id,...\n"
    assert payload["extra"]["brake_max_deg"] == 30.0
    assert "created_utc" in payload


def test_build_debug_dict_result_defaults(tmp_path):
    definition = _definition(tmp_path)
    data = build_debug_dict(
        mode="single",
        initial_definition=definition,
        target_curves=[],
        optimization_settings={},
        parameters=[],
        result={},
    )
    assert data["mode"] == "single"
    assert "optimized_mechanism" not in data
    assert "initial_mechanism_csv" not in data
