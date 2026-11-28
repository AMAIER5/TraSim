"""Regression tests for the GitHub issues #149 and #150.

- #149 "Relative angle between two coupled levers shall be
  same for braked and unbreaked position": levers 4 and 5 of
  the issue's mechanism are rigidly coupled by a shaft
  (``coupled`` column).  In the braked (pivot-shifted) build
  the re-solved partner angle was not propagated to the
  coupled lever, so the displayed relative angle changed
  between the unbraked and the braked position (148.15 deg
  vs 178.67 deg in the issue).  The builders now propagate
  the shaft coupling (constant offset from the reference
  pose) so the relative angle is identical in every build.
- #150 "Initial values not correct": the default (start)
  values of the optimization settings in the GUI sidebar
  must match the values shown in the issue's screenshot:
  population 110, max generations 200, children 100,
  mutation strength 0.035, stagnation limit 100, gradient
  weight 0.5, workers 0, seed 42, interpolated brake
  positions 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.curve_plotter import mechanism_states_from_results
from gui.pivot_shift_workflow import load_pivot_shift_inputs
from mechanics.pivot_shift_builder import PivotShiftBuilder
from model.mechanism_definition import (
    propagate_coupled_angles,
)
from simulation.mechanism_simulator import MechanismSimulator
from simulation.point_motion import PointMotion

REPO_ROOT = Path(__file__).parent.parent
APP_FILE = REPO_ROOT / "gui" / "app.py"
EXAMPLES = REPO_ROOT / "examples"

# Other GUI tests exec gui/app.py in-process with a shim that
# replaces st.file_uploader/st.line_chart; capture the real
# callables at import time (all test modules are imported
# before any test runs) so the AppTest below can restore them.
try:
    import streamlit as _st

    _REAL_FILE_UPLOADER = _st.file_uploader
    _REAL_LINE_CHART = _st.line_chart
except ImportError:  # pragma: no cover - streamlit missing
    _st = None
    _REAL_FILE_UPLOADER = None
    _REAL_LINE_CHART = None

# mechanism_initial.csv of issue #149 (semicolon CSV, German
# decimal comma; levers 4 and 5 are shaft-coupled via the
# ``coupled`` column, levers 2 and 3 share the shifted pivot).
ISSUE149_MECHANISM = """id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on;pivot_x_min;pivot_x_max;pivot_y_min;pivot_y_max;pivot_z_min;pivot_z_max
1;50,0;100,0;66,94792497874379;95,0;300,0;187,13233758975343;0,0;0,0;239,0;1,0;0,0;0,0;;;;;;;-20,0;20,0;-20,0;20,0;219,0;259,0
2;50,0;100,0;88,13767902419285;100,0;260,0;189,1408774958083;0,0;0,0;-261,0;1,0;0,0;0,0;;;;1;;;-1,0;1,0;-50,0;50,0;-311,0;-211,0
3;50,0;100,0;87,99237984163064;-90,0;90,0;-21,398470066263446;0,0;0,0;-261,0;1,0;0,0;0,0;;;;;2;;-1,0;1,0;-50,0;50,0;-311,0;-211,0
4;50,0;100,0;85,22135630641;-70,0;70,0;17,316936019894676;0,0;-40,0;-1020,0;1,0;0,0;0,0;;;;3;;;-20,0;20,0;-60,0;-20,0;-1040,0;-1000,0
5;50,0;100,0;97,98932661005541;59,99999999999999;300,0;165,46967901391216;-815,0;-40,0;-1020,0;1,0;0,0;0,0;;;;;4;;-835,0;-795,0;-60,0;-20,0;-1040,0;-1000,0
6;70,0;85,0;82,0;29,999999999999996;130,0;80,0;-850,0;-218,0;-4020,0;0,0;1,0;0,0;;;;5;;;-870,0;-830,0;-238,0;-198,0;-4040,0;-4000,0
""".replace(",", ".")

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


def _issue149_inputs():
    return load_pivot_shift_inputs(
        ISSUE149_MECHANISM,
        TARGET_CURVE,
        TARGET_CURVE,
    )


def _reference_angles(mechanism) -> dict[int, float]:
    return {
        index + 1: entry.reference_angle
        for index, entry in enumerate(mechanism.levers)
    }


# ----------------------------------------------------------------------
# Issue #149: constant relative angle of shaft-coupled levers
# ----------------------------------------------------------------------


def test_issue149_relative_angle_constant_across_builds():
    inputs = _issue149_inputs()
    definition = inputs.definition
    builder = PivotShiftBuilder(
        definition,
        shift_lever_id=inputs.shift_lever_id,
    )
    unbraked = builder.build((0.0, 0.0, 0.0))
    braked = builder.build((0.2, -0.5, 1.0))
    expected = (
        definition.get_lever(5).angle_start
        - definition.get_lever(4).angle_start
    )
    for mechanism in (unbraked, braked):
        refs = _reference_angles(mechanism)
        assert refs[5] - refs[4] == pytest.approx(
            expected,
            abs=1e-9,
        )
        # the stage that consumes the coupled lever uses an
        # input reference consistent with its partner's output
        stage_out_4 = next(
            stage.output_angle
            for stage in mechanism.stages
            if stage.output_lever is not None
            and _lever_id(mechanism, stage.output_lever) == 4
        )
        stage_in_5 = next(
            stage.input_angle
            for stage in mechanism.stages
            if _lever_id(mechanism, stage.input_lever) == 5
        )
        assert stage_in_5 - stage_out_4 == pytest.approx(
            expected,
            abs=1e-9,
        )
    assert (
        _reference_angles(unbraked)[5]
        - _reference_angles(unbraked)[4]
        == pytest.approx(
            _reference_angles(braked)[5]
            - _reference_angles(braked)[4],
            abs=1e-9,
        )
    )


def _lever_id(mechanism, lever) -> int:
    for index, entry in enumerate(mechanism.levers):
        if entry.lever is lever:
            return index + 1
    raise AssertionError("lever not found")


def test_issue149_displayed_angles_follow_shaft_coupling():
    inputs = _issue149_inputs()
    support = tuple(inputs.input_angles)
    offset = (
        inputs.definition.levers[0].angle_start
        - sum(support) / len(support)
    )
    simulator = MechanismSimulator(
        motion=PointMotion(
            angles=tuple(
                angle + offset for angle in support
            ),
        ),
    )
    relative = {}
    for label, shift in (
        ("unbraked", (0.0, 0.0, 0.0)),
        ("braked", (0.2, -0.5, 1.0)),
    ):
        builder = PivotShiftBuilder(
            inputs.definition,
            shift_lever_id=inputs.shift_lever_id,
        )
        mechanism = builder.build(shift)
        results = simulator.simulate(mechanism)
        assert all(
            result.success for result in results
        )
        states = mechanism_states_from_results(
            mechanism,
            results,
        )
        for state in states:
            angles = {
                lever.name: lever.angle
                for lever in state.levers
            }
            relative.setdefault(label, []).append(
                angles["Lever 5"] - angles["Lever 4"]
            )
    for label, values in relative.items():
        assert len(values) == 3
        assert max(values) - min(values) < 1e-9, label
    assert relative["braked"] == pytest.approx(
        relative["unbraked"],
        abs=1e-9,
    )


def test_propagate_coupled_angles_skips_driven_and_root():
    inputs = _issue149_inputs()
    definition = inputs.definition
    angles = {
        lever.id: lever.angle_start
        for lever in definition.levers
    }
    # simulate a re-solved partner: lever 4 moved by 0.3 rad
    angles[4] = angles[4] + 0.3
    propagate_coupled_angles(
        definition,
        angles,
        skip_ids=(1,),
    )
    expected = (
        definition.get_lever(5).angle_start
        - definition.get_lever(4).angle_start
    )
    assert angles[5] == pytest.approx(angles[4] + expected)
    # driven levers keep their (solved) angle
    assert angles[4] == pytest.approx(
        definition.get_lever(4).angle_start + 0.3
    )
    # the root drive lever is never overwritten
    assert angles[1] == pytest.approx(
        definition.get_lever(1).angle_start
    )


# ----------------------------------------------------------------------
# Issue #150: default (start) values of the optimization settings
# ----------------------------------------------------------------------

EXPECTED_DEFAULTS = {
    "Populationsgr\u00f6\u00dfe": 110,
    "Maximale Generationen": 200,
    "Kinder pro Generation": 100,
    "Stagnationslimit (Generationen)": 100,
    "Worker-Prozesse": 0,
    "Zufallsseed": 42,
    "Interpolierte Bremsstellungen": 0,
}
EXPECTED_SLIDERS = {
    "Mutationsst\u00e4rke": 0.035,
    "Gradientengewicht": 0.5,
}


def test_issue150_default_settings_match_screenshot():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    if _REAL_FILE_UPLOADER is not None:
        _st.file_uploader = _REAL_FILE_UPLOADER
        _st.line_chart = _REAL_LINE_CHART

    app = AppTest.from_file(str(APP_FILE))
    app.run()
    app.file_uploader[0].set_value(
        (
            "mechanism.csv",
            (EXAMPLES / "mechanism.csv").read_bytes(),
            "text/csv",
        )
    )
    app.file_uploader[1].set_value(
        (
            "targetcurve.csv",
            (EXAMPLES / "target_curve.csv").read_bytes(),
            "text/csv",
        )
    )
    app.run()
    assert not app.exception

    for number_input in app.sidebar.number_input:
        if number_input.label in EXPECTED_DEFAULTS:
            assert number_input.value == (
                EXPECTED_DEFAULTS[number_input.label]
            ), number_input.label
    for slider in app.sidebar.slider:
        if slider.label in EXPECTED_SLIDERS:
            assert slider.value == pytest.approx(
                EXPECTED_SLIDERS[slider.label]
            ), slider.label
    # every expected field was actually found
    found_numbers = {
        number_input.label
        for number_input in app.sidebar.number_input
    }
    found_sliders = {
        slider.label for slider in app.sidebar.slider
    }
    assert set(EXPECTED_DEFAULTS) <= found_numbers
    assert set(EXPECTED_SLIDERS) <= found_sliders


def test_issue150_default_settings_in_source():
    """Source-level guard for the start values (issue #150):
    the sidebar defaults in gui/app.py must match the
    screenshot of the issue."""
    source = APP_FILE.read_text(encoding="utf-8")
    assert 'value=110,\n        step=10,' in source
    assert 'value=0.035,\n        step=0.001,' in source
    assert (
        '"Interpolierte Bremsstellungen",\n'
        "        min_value=0,\n"
        "        max_value=20,\n"
        "        value=0,"
    ) in source
