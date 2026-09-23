"""tests/test_curve_plotter.py

Tests for the HTML visualization module.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from analysis.curve_plotter import (
    CurvePlotter,
    LeverPlotLine,
    MechanismPlotState,
    mechanism_states_from_results,
    target_input_angles,
)
from analysis.target_curve import TargetCurve
from core.point3d import Point3D
from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)
from mechanism_io.csv_reader import CsvReader
from optimization.parameter_set import (
    ParameterSet,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.point_motion import PointMotion
from simulation.stage_simulator import (
    StageSimulator,
)


def _build_mechanism(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )
    return (
        CsvMechanismBuilder(definition).build(
            ParameterSet(()),
        ),
        definition,
    )


def test_lever_plot_line_stores_geometry():
    lever = LeverPlotLine(
        name="Lever 1",
        pivot=Point3D(0.0, 0.0, 0.0),
        end=Point3D(10.0, 0.0, 0.0),
        angle=math.radians(90),
        color="#000000",
    )
    assert lever.pivot.x == 0.0
    assert lever.end.x == 10.0
    assert lever.name == "Lever 1"


def test_mechanism_state_from_mechanism(
    example_mechanism_csv,
):
    mechanism, _ = _build_mechanism(
        example_mechanism_csv,
    )
    state = (
        MechanismPlotState.from_mechanism(
            mechanism,
            lever_angles={1: math.radians(30)},
            position_label="min",
            position_angle=math.radians(30),
        )
    )
    assert state.position_label == "min"
    assert len(state.levers) == 4
    assert math.degrees(
        state.levers[0].angle,
    ) == pytest.approx(30)
    assert state.levers[0].name == "Lever 1"
    assert state.levers[0].pivot == (
        Point3D(0.0, 0.0, 0.0)
    )


def test_mechanism_state_uses_reference_angles(
    example_mechanism_csv,
):
    mechanism, _ = _build_mechanism(
        example_mechanism_csv,
    )
    state = (
        MechanismPlotState.from_mechanism(
            mechanism,
            lever_angles={},
        )
    )
    assert all(
        lever.end.x > 0 or lever.end.y > 0
        for lever in state.levers
    )


def test_states_from_results_three_positions(
    simple_stage_csv,
):
    definition = CsvReader.read_mechanism(
        simple_stage_csv,
    )
    mechanism = CsvMechanismBuilder(
        definition,
    ).build(ParameterSet(()))
    motion = PointMotion(
        angles=(
            math.radians(-40),
            math.radians(0),
            math.radians(40),
        ),
    )
    simulator = MechanismSimulator(
        motion=motion,
        stage_simulator=StageSimulator(),
    )
    results = simulator.simulate(mechanism)
    assert all(
        result.success for result in results
    )
    min_state, mid_state, max_state = (
        mechanism_states_from_results(
            mechanism,
            results,
        )
    )
    labels = (
        min_state.position_label,
        mid_state.position_label,
        max_state.position_label,
    )
    assert labels == ("min", "mid", "max")
    assert math.degrees(
        min_state.position_angle,
    ) == pytest.approx(-40)
    assert math.degrees(
        mid_state.position_angle,
    ) == pytest.approx(0)
    assert math.degrees(
        max_state.position_angle,
    ) == pytest.approx(40)
    assert len(min_state.levers) == 2
    assert len(mid_state.levers) == 2
    assert len(max_state.levers) == 2
    assert min_state.levers[1].angle == (
        pytest.approx(
            results[0].output_angles[0],
        )
    )


def test_states_from_results_requires_samples(
    example_mechanism_csv,
):
    try:
        mechanism_states_from_results(
            _build_mechanism(
                example_mechanism_csv,
            )[0],
            (),
        )
        assert False
    except ValueError:
        assert True


def test_plotter_build_data_degrees():
    plotter = CurvePlotter(title="Test")
    plotter.set_target_curve(
        TargetCurve.from_points(
            (math.radians(0), math.radians(90)),
            (math.radians(10), math.radians(80)),
        ),
    )
    plotter.add_actual_curve(
        (math.radians(0), math.radians(90)),
        (math.radians(12), math.radians(82)),
    )
    data = plotter.build_data()
    assert data["title"] == "Test"
    assert len(data["curves"]) == 2
    target_curve = data["curves"][0]
    assert target_curve["label"] == (
        "Soll-Ausgangskurve"
    )
    assert target_curve["output_angles"] == [
        10.0,
        80.0,
    ]
    actual_curve = data["curves"][1]
    assert actual_curve["label"] == (
        "Ist-Ausgangskurve"
    )
    assert actual_curve["output_angles"] == [
        pytest.approx(12.0),
        pytest.approx(82.0),
    ]


def test_target_input_angles_extracts_supports():
    target = TargetCurve.from_points(
        (math.radians(0), math.radians(90)),
        (math.radians(10), math.radians(80)),
    )
    angles = target_input_angles(target)
    assert math.degrees(angles[0]) == 0
    assert math.degrees(angles[1]) == 90


def test_set_target_curve_with_explicit_points():
    plotter = CurvePlotter()
    plotter.set_target_curve(
        TargetCurve.from_points(
            (math.radians(0), math.radians(90)),
            (math.radians(10), math.radians(80)),
        ),
        input_angles=(
            math.radians(0),
            math.radians(90),
        ),
        output_angles=(
            math.radians(10),
            math.radians(80),
        ),
    )
    data = plotter.build_data()
    assert data["curves"][0][
        "output_angles"
    ] == [10.0, 80.0]


def test_build_html_embeds_json():
    plotter = CurvePlotter(title="Test")
    plotter.set_target_curve(
        TargetCurve.from_points(
            (math.radians(0), math.radians(90)),
            (math.radians(10), math.radians(80)),
        ),
    )
    html = plotter.build_html()
    match = re.search(
        r'<script id="plot-data" '
        r'type="application/json">'
        r"(.*?)"
        r"</script>",
        html,
        re.S,
    )
    assert match is not None
    data = json.loads(match.group(1))
    assert data["title"] == "Test"
    assert len(data["curves"]) == 1


def test_write_creates_file(tmp_path):
    output = tmp_path / "result.html"
    plotter = CurvePlotter(title="Test")
    plotter.set_target_curve(
        TargetCurve.from_points(
            (math.radians(0), math.radians(90)),
            (math.radians(10), math.radians(80)),
        ),
    )
    path = plotter.write(output)
    assert path == output
    assert output.exists()
    content = output.read_text(
        encoding="utf-8",
    )
    assert "<!DOCTYPE html>" in content
    assert "lever-plot" in content
    assert "curve-plot" in content


def test_build_html_escapes_title():
    plotter = CurvePlotter(
        title='<script>"x"</script>',
    )
    html = plotter.build_html()
    assert "<script>" not in html.replace(
        '<script id="plot-data"',
        "",
    ).replace(
        '<script src="',
        "",
    ).replace(
        "<script>",
        "",
    )


def test_lever_colors_cycled(
    example_mechanism_csv,
):
    mechanism, _ = _build_mechanism(
        example_mechanism_csv,
    )
    state = (
        MechanismPlotState.from_mechanism(
            mechanism,
            lever_angles={},
        )
    )
    colors = [
        lever.color
        for lever in state.levers
    ]
    assert len(set(colors)) == 4


def test_mechanism_state_contains_rods(
    example_mechanism_csv,
):
    mechanism, _ = _build_mechanism(
        example_mechanism_csv,
    )
    state = (
        MechanismPlotState.from_mechanism(
            mechanism,
            lever_angles={},
        )
    )
    assert len(state.rods) == 3
    assert state.rods[0].name == "Koppelstange 1"
    for rod in state.rods:
        assert rod.length > 0


def test_rods_connect_lever_endpoints(
    example_mechanism_csv,
):
    mechanism, _ = _build_mechanism(
        example_mechanism_csv,
    )
    state = (
        MechanismPlotState.from_mechanism(
            mechanism,
            lever_angles={},
        )
    )
    levers = state.levers
    rods = state.rods
    assert rods[0].start == levers[0].end
    assert rods[0].end == levers[1].end
    assert rods[1].start == levers[1].end
    assert rods[1].end == levers[2].end


def test_build_data_contains_rods(
    example_mechanism_csv,
):
    mechanism, _ = _build_mechanism(
        example_mechanism_csv,
    )
    plotter = CurvePlotter()
    plotter.add_mechanism_state(
        MechanismPlotState.from_mechanism(
            mechanism,
            lever_angles={},
        )
    )
    data = plotter.build_data()
    rods = data["mechanism_states"][0]["rods"]
    assert len(rods) == 3
    assert rods[0]["name"] == "Koppelstange 1"
    assert "length" in rods[0]
