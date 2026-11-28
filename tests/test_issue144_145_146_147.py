"""Regression tests for the GitHub issues #144-#147.

- #144 "Optimized parameters are not displayed and can't be
  overtaken for next cycle": after a pivot-shift run only
  ``pivot_shift_result`` is set in the session state, but the
  "Parameter des Startmechanismus" table built its optimized
  map only from ``result`` / ``mixed_brake_result``, so every
  "Optimiert" cell stayed ``None`` and nothing could be
  adopted.  Covered with a real headless app run (AppTest)
  plus the shift-parameter wiring.
- #145 "Pivot shifting not working in z-direction": the pair
  pivot shift was planar ``(dx, dy)`` only; the z axis of the
  optional pivot limits was ignored.  The shift is now spatial
  ``(dx, dy, dz)`` with a ``shift.z`` parameter clamped to the
  z limits (mechanism_initial.csv attached to the issue).
- #146 "Diagram area not completely visible without
  scrolling": the report iframe must be tall enough for the
  whole HTML document (transfer curves AND 3D mechanism).
- #147 "Mechanism diagram shall show largest axis
  horizontally": the initial 3D camera puts the axis with the
  largest min-max span horizontally on screen.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from analysis.brake_curve_fitness import BrakeCurveFitness
from analysis.curve_plotter import (
    CurvePlotter,
    LeverPlotLine,
    MechanismPlotState,
)
from core.point3d import Point3D
from gui.pivot_shift_workflow import (
    PivotShiftSettings,
    load_pivot_shift_inputs,
    pivot_shift_parameter_template,
    run_pivot_shift_optimization,
)
from mechanics.pivot_shift_builder import (
    PivotShiftBuilder,
    normalize_shift,
)
from optimization.parameter_set import ParameterSet
from optimization.pivot_shift_optimizer import (
    PivotShiftOptimizer,
    shift_parameter_names,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.point_motion import PointMotion

REPO_ROOT = Path(__file__).parent.parent
APP_FILE = REPO_ROOT / "gui" / "app.py"

# Other GUI tests (e.g. tests/test_issue135_136_pivot_shift_gui.py)
# exec gui/app.py in-process with a shim that replaces
# ``st.file_uploader``/``st.line_chart``; capture the real
# callables at import time (all test modules are imported
# before any test runs) so this test can restore them.
try:
    import streamlit as _st

    _REAL_FILE_UPLOADER = _st.file_uploader
    _REAL_LINE_CHART = _st.line_chart
except ImportError:  # pragma: no cover - streamlit missing
    _st = None
    _REAL_FILE_UPLOADER = None
    _REAL_LINE_CHART = None

# mechanism_initial.csv attached to issue #145 (semicolon CSV,
# German decimal comma; levers 2 and 3 share the pivot
# (0, 0, -261) and carry per-axis pivot limits incl. z).
ISSUE145_MECHANISM = """id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on;pivot_x_min;pivot_x_max;pivot_y_min;pivot_y_max;pivot_z_min;pivot_z_max
1;50;100;73,05799607;95;300;133,651956;0;0;239;1;0;0;;;;;;;;;;;;
2;50;100;76,9929424;100;260;153,1432978;0;0;-261;1;0;0;;;;1;;;-1;1;-50;50;-311;-211
3;50;100;91,73216337;-90;90;11,8383644;0;0;-261;1;0;0;;;;;2;;-1;1;-50;50;-311;-211
4;50;100;62,09694189;-70;70;-5,795825305;0;-40;-1020;1;0;0;;;;3;;;;;;;;
5;50;100;92,07312009;60;300;147,4731758;-815;-40;-1020;1;0;0;;;;;4;;;;;;;
6;70;85;82;30;130;80;-850;-218;-4020;0;1;0;;;;5;;;;;;;;
"""

TARGET_CURVE = """input_angle;output_angle
-35;-25
-30;-23
-25;-20
-20;-17
-10;-9
0;0
10;12
20;24
25;31
30;38
35;46
"""


def _issue145_inputs():
    return load_pivot_shift_inputs(
        ISSUE145_MECHANISM,
        TARGET_CURVE,
        TARGET_CURVE,
    )


def _with_values(template, **overrides):
    parameters = [
        (
            replace(
                parameter,
                value=overrides[parameter.name],
            )
            if parameter.name in overrides
            else parameter
        )
        for parameter in template.parameters
    ]
    return ParameterSet(parameters=tuple(parameters))


def _optimizer(inputs):
    return PivotShiftOptimizer(
        builder=PivotShiftBuilder(
            inputs.definition,
            shift_lever_id=inputs.shift_lever_id,
        ),
        simulator=MechanismSimulator(
            motion=PointMotion(
                angles=tuple(inputs.input_angles),
            ),
        ),
        fitness=BrakeCurveFitness(
            weights=(1.0, 1.0),
            max_weight=1.0,
        ),
        definition=inputs.definition,
        targets=(inputs.unbraked, inputs.braked),
        shift_lever_id=inputs.shift_lever_id,
    )


# ----------------------------------------------------------------------
# Issue #145: the pivot shift works in the z direction as well
# ----------------------------------------------------------------------


def test_normalize_shift_pads_to_three_components():
    assert normalize_shift(()) == (0.0, 0.0, 0.0)
    assert normalize_shift((1.0, 2.0)) == (1.0, 2.0, 0.0)
    assert normalize_shift((1.0, 2.0, 3.0)) == (
        1.0,
        2.0,
        3.0,
    )
    with pytest.raises(ValueError):
        normalize_shift((1.0,))


def test_builder_applies_three_component_shift():
    inputs = _issue145_inputs()
    builder = PivotShiftBuilder(
        inputs.definition,
        shift_lever_id=inputs.shift_lever_id,
    )
    braked = builder.build((3.0, -4.0, 25.0))
    lever_ids = [
        lever.id for lever in inputs.definition.levers
    ]
    pivots = {
        lever_ids[index]: entry.lever.pivot
        for index, entry in enumerate(braked.levers)
    }
    # the pair (levers 2 and 3) moved on every axis
    assert pivots[2].x == pytest.approx(3.0)
    assert pivots[2].y == pytest.approx(-4.0)
    assert pivots[2].z == pytest.approx(-261.0 + 25.0)
    assert pivots[3].z == pytest.approx(-261.0 + 25.0)
    # other levers keep their pivot
    assert pivots[1].z == pytest.approx(239.0)
    # a two-component shift leaves z untouched
    planar = builder.build((3.0, -4.0))
    planar_pivots = {
        lever_ids[index]: entry.lever.pivot
        for index, entry in enumerate(planar.levers)
    }
    assert planar_pivots[2].z == pytest.approx(-261.0)


def test_factory_creates_shift_z_parameter_from_z_limits():
    inputs = _issue145_inputs()
    template = pivot_shift_parameter_template(inputs)
    names = {p.name for p in template.parameters}
    assert names >= set(shift_parameter_names(2))
    z = template.get("lever.2.shift.z")
    # pivot z = -261 with limits [-311, -211]
    assert (z.minimum, z.maximum) == (-50.0, 50.0)
    assert z.value == 0.0


def test_optimizer_clamps_shift_in_z_direction():
    inputs = _issue145_inputs()
    template = pivot_shift_parameter_template(inputs)
    optimizer = _optimizer(inputs)
    # base pivot moved to z = -220: the room left to the
    # z_max limit (-211) is a shift of +9 although the raw
    # shift.z parameter asks for +50
    candidate = _with_values(
        template,
        **{
            "lever.2.pivot.z": -220.0,
            "lever.2.shift.x": 0.0,
            "lever.2.shift.y": 0.0,
            "lever.2.shift.z": 50.0,
        },
    )
    shift = optimizer.shift_for(candidate)
    assert shift == (0.0, 0.0, 9.0)
    mechanism = optimizer._cached_mechanism(
        candidate,
        shift,
    )
    assert mechanism is not None
    braked_pivot = mechanism.levers[1].lever.pivot
    assert braked_pivot.z == pytest.approx(-211.0)


def test_short_pivot_run_reports_3d_shift():
    inputs = _issue145_inputs()
    template = pivot_shift_parameter_template(inputs)
    settings = PivotShiftSettings(
        population_size=10,
        children_count=8,
        selection_count=4,
        max_generations=3,
        stagnation_limit=5,
        workers=1,
        seed=42,
    )
    result = run_pivot_shift_optimization(
        inputs,
        template,
        settings,
    )
    assert result.best_score < float("inf")
    assert len(result.shift_mm) == 3
    assert "lever.2.shift.z" in result.best_values


# ----------------------------------------------------------------------
# Issue #144: optimized values are shown after a pivot-shift run
# ----------------------------------------------------------------------


def test_issue144_app_fills_optimized_columns_after_pivot_run():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    if _REAL_FILE_UPLOADER is not None:
        _st.file_uploader = _REAL_FILE_UPLOADER
        _st.line_chart = _REAL_LINE_CHART

    def uploader(key: str):
        return next(
            widget
            for widget in app.file_uploader
            if widget.key == key
        )

    app = AppTest.from_file(
        str(APP_FILE),
        default_timeout=300,
    )
    app.run()
    uploader("mechanism").set_value(
        (
            "mechanism_initial.csv",
            ISSUE145_MECHANISM.encode("utf-8"),
            "text/csv",
        )
    )
    uploader("target").set_value(
        (
            "targetcurve.csv",
            TARGET_CURVE.encode("utf-8"),
            "text/csv",
        )
    )
    app.run()
    assert not app.exception
    # the braked-curve picker is only enabled once the
    # mechanism upload revealed a lever pair
    uploader("target_braked").set_value(
        (
            "target_curve_braked.csv",
            TARGET_CURVE.encode("utf-8"),
            "text/csv",
        )
    )
    app.run()
    assert not app.exception
    assert app.session_state.get("pivot_shift_result") is None

    sidebar_values = {
        "Populationsgr\u00f6\u00dfe": 10,
        "Maximale Generationen": 3,
        "Kinder pro Generation": 8,
        "Stagnationslimit (Generationen)": 5,
        "Worker-Prozesse": 1,
    }
    for number_input in app.sidebar.number_input:
        if number_input.label in sidebar_values:
            number_input.set_value(
                sidebar_values[number_input.label]
            )
    app.run()
    assert not app.exception

    start = next(
        button
        for button in app.button
        if button.key == "start_optimization_button"
    )
    start.click().run()
    assert not app.exception
    assert (
        app.session_state.get("pivot_shift_result")
        is not None
    )

    editor = next(
        candidate
        for candidate in (
            list(getattr(app, "data_editor", None) or [])
            or list(app.dataframe)
        )
        if str(candidate.key).startswith("mechanism_editor")
    )
    optimized = editor.value
    # Issue #144: the "Optimiert" columns must show the values
    # of the finished pivot-shift run instead of None.
    assert any(
        value is not None
        for value in optimized["length_optimized"]
    )
    assert any(
        value is not None
        for value in optimized["angle_optimized"]
    )


def test_issue144_app_source_reads_pivot_result_for_optimized_map():
    """Source-level guard: the optimized-values map of the
    start-mechanism table must also be built from
    ``pivot_shift_result`` (gui/app.py cannot be imported)."""
    source = APP_FILE.read_text(encoding="utf-8")
    assert "pivot_shift_result" in source
    match = re.search(
        r"pivot_result_state\s*=\s*st\.session_state\.get\(\s*"
        r'"pivot_shift_result"',
        source,
    )
    assert match is not None
    assert re.search(
        r"optimized_values\(\s*"
        r"pivot_result_state\s*\.\s*optimized_definition",
        source,
    )


# ----------------------------------------------------------------------
# Issue #146: the whole report fits into the diagram area
# ----------------------------------------------------------------------


def test_issue146_report_iframe_covers_whole_document():
    source = APP_FILE.read_text(encoding="utf-8")
    match = re.search(
        r"REPORT_IFRAME_HEIGHT\s*=\s*(\d+)",
        source,
    )
    assert match is not None
    iframe_height = int(match.group(1))
    # The HTML report stacks the 2D curve plot (~560 px plus
    # chrome) and the 3D mechanism plot (~720 px plus headers,
    # margins and containers); the iframe must be at least as
    # tall as the whole document so nothing is cut off.
    assert iframe_height >= 1600
    assert "height=900" not in source
    assert source.count("st.iframe(") == source.count(
        "height=REPORT_IFRAME_HEIGHT"
    )

    plotter_source = (
        REPO_ROOT / "analysis" / "curve_plotter.py"
    ).read_text(encoding="utf-8")
    lever = re.search(
        r"#lever-plot\s*\{[^}]*height:\s*(\d+)px",
        plotter_source,
        re.DOTALL,
    )
    curve = re.search(
        r"#curve-plot\s*\{[^}]*height:\s*(\d+)px",
        plotter_source,
        re.DOTALL,
    )
    assert lever is not None and int(lever.group(1)) >= 700
    assert curve is not None and int(curve.group(1)) >= 540
    # iframe height must cover both plots plus the page chrome
    assert iframe_height >= (
        int(lever.group(1)) + int(curve.group(1)) + 400
    )


# ----------------------------------------------------------------------
# Issue #147: largest axis horizontal in the initial 3D view
# ----------------------------------------------------------------------


def _axis_of(vector: dict) -> str:
    nonzero = [
        axis
        for axis in ("x", "y", "z")
        if vector.get(axis, 0.0) != 0.0
    ]
    assert len(nonzero) == 1
    return nonzero[0]


def test_issue147_largest_span_axis_is_horizontal():
    plotter = CurvePlotter(title="issue 147")
    # spans: x = 10, y = 20, z = 4000  ->  z must be horizontal
    plotter.add_mechanism_state(
        MechanismPlotState(
            position_label="min",
            position_angle=0.0,
            levers=(
                LeverPlotLine(
                    name="Lever 1",
                    pivot=Point3D(0.0, 0.0, 0.0),
                    end=Point3D(0.0, 0.0, 4000.0),
                    angle=0.0,
                    color="#1f77b4",
                ),
                LeverPlotLine(
                    name="Lever 2",
                    pivot=Point3D(0.0, 0.0, 0.0),
                    end=Point3D(10.0, 20.0, 0.0),
                    angle=0.0,
                    color="#d62728",
                ),
            ),
        )
    )
    camera = plotter.build_data()["initial_camera"]
    eye_axis = _axis_of(camera["eye"])
    up_axis = _axis_of(camera["up"])
    # the largest-span axis (z) is neither the view direction
    # (middle span y) nor the screen-up direction (smallest
    # span x), so it lies horizontally on screen
    assert eye_axis == "y"
    assert up_axis == "x"
    html = plotter.build_html()
    assert "data.initial_camera" in html
    assert "camera: camera" in html


def test_issue147_default_view_for_equal_spans():
    plotter = CurvePlotter(title="issue 147 default")
    plotter.add_mechanism_state(
        MechanismPlotState(
            position_label="min",
            position_angle=0.0,
            levers=(
                LeverPlotLine(
                    name="Lever 1",
                    pivot=Point3D(0.0, 0.0, 0.0),
                    end=Point3D(100.0, 100.0, 100.0),
                    angle=0.0,
                    color="#1f77b4",
                ),
            ),
        )
    )
    camera = plotter.build_data()["initial_camera"]
    # all spans equal: conventional view (x horizontal, z up)
    assert _axis_of(camera["eye"]) == "y"
    assert _axis_of(camera["up"]) == "z"
