"""Regression tests for GitHub issue #141

"Optimization is inaccurate, weighting is not considered
properly" (pivot shift mode, with the mechanism_initial.csv
and the unbraked/braked target curves attached to the
issue):

- Stale/dangling lever references must be rejected: a
  removed lever (e.g. the old brake lever 7) must not
  leave behind a ``pivot_on`` / ``driver`` / ``coupled``
  reference on the remaining levers.
- The mechanism editor must carry over the per-axis
  pivot limits of the shift lever; otherwise the
  ±50 mm shift room from the CSV silently degrades to
  the ±20 mm default after a table edit.
- The pivot shift fitness must compare the simulated
  output against the target curve in the ANCHORED output
  frame (same convention as the mixed brake mode).  With
  the unanchored output lever the whole Ist curve carried
  the constant installation offset of the output lever,
  so no per-point weighting could visibly shape the
  result (the 0 deg / 0 deg point weighted with 100 had
  no effect).
- The weights of the currently loaded target files
  must reach the shared two-curve editor and the weight
  chart after a rerun (also when the same cell carried
  an editor edit in the previous run).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gui.pivot_shift_workflow import (
    PivotShiftSettings,
    load_pivot_shift_inputs,
    pivot_shift_parameter_template,
    pivot_shift_result_report,
    run_pivot_shift_optimization,
    shift_room_box,
)
from gui.workflow import (
    WorkflowError,
    with_mechanism_levers,
)
from model.mechanism_definition import (
    MechanismDefinition,
)

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)

MECHANISM = (
    EXAMPLES_DIR
    / "pivot_shift_mechanism.csv"
).read_text(encoding="utf-8")
UNBRAKED = (
    EXAMPLES_DIR
    / "target_curve_unbraked.csv"
).read_text(encoding="utf-8")
BRAKED = (
    EXAMPLES_DIR
    / "target_curve_braked.csv"
).read_text(encoding="utf-8")

ISSUE_MECHANISM = """id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on;pivot_x_min;pivot_x_max;pivot_y_min;pivot_y_max
1;50;100;73,05799607;95;300;133,651956;0;0;239;1;0;0;;;;;;;;;;
2;50;100;76,9929424;100;260;153,1432978;0;0;-261;1;0;0;;;;1;;;-50;50;-50;50
3;50;100;91,73216337;-90;90;11,8383644;0;0;-261;1;0;0;;;;;2;;-50;50;-50;50
4;50;100;62,09694189;-70;70;-5,795825305;0;-40;-1020;1;0;0;;;;3;;;;;;
5;50;100;92,07312009;60;300;147,4731758;-815;-40;-1020;1;0;0;;;;;4;;;;;
6;70;85;82;30;130;80;-850;-218;-4020;0;1;0;;;;5;;;;;;
"""

ISSUE_UNBRAKED = """input_angle;output_angle
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


def _load_mechanism_inputs() -> "object":  # noqa: ANN202
    from gui.workflow import load_inputs

    return load_inputs(MECHANISM, UNBRAKED)


def _small_run(inputs, generations: int = 8):
    template = pivot_shift_parameter_template(
        inputs
    )
    settings = PivotShiftSettings(
        population_size=30,
        children_count=30,
        max_generations=generations,
        stagnation_limit=generations,
        workers=1,
        seed=42,
    )
    return run_pivot_shift_optimization(
        inputs,
        template,
        settings,
    )


def test_definition_rejects_dangling_pivot_on():
    """Issue #141 (1): a lever that referenced a removed
    lever (pivot_on "7" while no lever 7 is defined) must
    be rejected instead of silently carrying the stale
    reference into the optimization and the debug
    export."""
    from core.point3d import Point3D
    from model.lever_definition import (
        LeverDefinition,
    )
    from core.vector3d import Vector3D

    _AXIS = Vector3D(x=0.0, y=0.0, z=1.0)

    def lever(lever_id: int, **kwargs):
        defaults = dict(
            id=lever_id,
            pivot=Point3D(x=0.0, y=0.0, z=0.0),
            length_min=50.0,
            length_max=100.0,
            length_start=70.0,
            angle_min=math.radians(0.0),
            angle_max=math.radians(100.0),
            angle_start=math.radians(10.0),
            axis=_AXIS,
        )
        defaults.update(kwargs)
        return LeverDefinition(**defaults)

    with pytest.raises(ValueError, match="pivot_on"):
        MechanismDefinition(
            (
                lever(1),
                lever(2, pivot_on="7"),
            )
        )

    with pytest.raises(ValueError, match="driver"):
        MechanismDefinition(
            (
                lever(1),
                lever(2, driver=9),
            )
        )

    with pytest.raises(ValueError, match="coupled"):
        MechanismDefinition(
            (
                lever(1),
                lever(2, coupled=9),
            )
        )

    # valid references keep working
    definition = MechanismDefinition(
        (lever(1), lever(2, driver=1))
    )
    assert definition.lever_count == 2


def test_mechanism_editor_keeps_pivot_limits():
    """Issue #141 (1): the lever table must not drop the
    per-axis pivot limits; otherwise the shift room of the
    issue's mechanism (±50 mm) silently degrades to the
    ±20 mm default after any table edit."""
    inputs = _load_mechanism_inputs()
    lever = inputs.definition.levers[0]
    rows = [
        {
            "id": row.id,
            "length_min": row.length_min,
            "length_max": row.length_max,
            "length": row.length_start,
            "angle_min": math.degrees(
                row.angle_min
            ),
            "angle_max": math.degrees(
                row.angle_max
            ),
            "angle": math.degrees(
                row.angle_start
            ),
            "pivot_x": row.pivot.x,
            "pivot_y": row.pivot.y,
            "pivot_z": row.pivot.z,
            "axis_x": row.axis.x,
            "axis_y": row.axis.y,
            "axis_z": row.axis.z,
            "driver": row.driver,
            "coupled": row.coupled,
            "pivot_x_min": row.pivot_x_min,
            "pivot_x_max": row.pivot_x_max,
            "pivot_y_min": row.pivot_y_min,
            "pivot_y_max": row.pivot_y_max,
            "pivot_z_min": row.pivot_z_min,
            "pivot_z_max": row.pivot_z_max,
        }
        for row in inputs.definition.levers
    ]
    edited = with_mechanism_levers(inputs, rows)
    del lever
    for original, edited_lever in zip(
        inputs.definition.levers,
        edited.definition.levers,
    ):
        assert (
            edited_lever.pivot_x_min
            == original.pivot_x_min
        )
        assert (
            edited_lever.pivot_x_max
            == original.pivot_x_max
        )
        assert (
            edited_lever.pivot_y_min
            == original.pivot_y_min
        )
        assert (
            edited_lever.pivot_y_max
            == original.pivot_y_max
        )


def test_pivot_run_uses_shift_room_from_csv():
    """The ±50 mm shift room of the issue's mechanism must
    reach the optimization (previously degraded to ±20
    after the GUI mechanism table round trip)."""
    inputs = load_pivot_shift_inputs(
        ISSUE_MECHANISM,
        ISSUE_UNBRAKED,
        ISSUE_UNBRAKED,
    )
    template = pivot_shift_parameter_template(
        inputs
    )
    bounds = {
        parameter.name: (
            parameter.minimum,
            parameter.maximum,
        )
        for parameter in template.parameters
    }
    assert bounds["lever.2.shift.x"] == (-50.0, 50.0)
    assert bounds["lever.2.shift.y"] == (-50.0, 50.0)


def test_pivot_run_anchors_output_frame():
    """Issue #141 (3): the optimized mechanism's actual
    curve must follow the target curve in the anchored
    output frame (no constant installation offset of the
    output lever), so the per-point weights actually
    shape the result; the middle support point of the
    unbraked curve is hit exactly."""
    inputs = load_pivot_shift_inputs(
        ISSUE_MECHANISM,
        ISSUE_UNBRAKED,
        ISSUE_UNBRAKED,
    )
    result = _small_run(inputs)
    assert result.unbraked_actual_deg

    middle = len(result.input_angles_deg) // 2
    assert abs(
        result.unbraked_actual_deg[middle]
        - result.unbraked_target_deg[middle]
    ) < 1e-6

    deviations = [
        abs(actual - target)
        for actual, target in zip(
            result.unbraked_actual_deg,
            result.unbraked_target_deg,
        )
    ]
    assert max(deviations) < 5.0


def test_result_report_shows_mechanism_and_room():
    """Issue #140/#141: the result report must contain the
    mechanism snapshots of the unbraked AND the braked
    position plus the shift room box."""
    braked_lines = []
    for line in ISSUE_UNBRAKED.splitlines():
        if not line.strip():
            continue
        parts = line.split(";")
        if len(parts) < 2:
            continue
        try:
            parts[1] = str(
                float(parts[1].replace(",", "."))
                - 4.0
            )
        except ValueError:
            pass
        braked_lines.append(";".join(parts))
    inputs = load_pivot_shift_inputs(
        ISSUE_MECHANISM,
        ISSUE_UNBRAKED,
        "\n".join(braked_lines) + "\n",
    )
    result = _small_run(inputs)
    assert result.html
    payload = json.loads(
        result.html.split(
            'type="application/json">'
        )[1].split("</script>")[0]
    )
    labels = {
        series["label"]
        for series in payload["curves"]
    }
    assert {
        "Soll ungebremst",
        "Soll voll gebremst",
        "Ist ungebremst",
        "Ist voll gebremst",
    } <= labels
    # three snapshots per position
    assert len(payload["mechanism_states"]) == 6
    braked = [
        state
        for state in payload["mechanism_states"]
        if any(
            "gebremst" in lever["name"]
            for lever in state["levers"]
        )
    ]
    assert len(braked) == 3
    boxes = payload["boxes"]
    assert len(boxes) == 1
    assert boxes[0]["min"]["x"] == -50.0
    assert boxes[0]["max"]["x"] == 50.0


def test_shift_room_box_uses_pivot_limits_and_default():
    inputs = load_pivot_shift_inputs(
        MECHANISM,
        UNBRAKED,
        BRAKED,
    )
    lever = inputs.definition.get_lever(
        inputs.shift_lever_id
    )
    box = shift_room_box(
        inputs.definition,
        inputs.shift_lever_id,
    )
    # per-axis limits from the CSV bound the room
    assert box.minimum.x == lever.pivot_x_min
    assert box.maximum.x == lever.pivot_x_max
    assert box.minimum.y == lever.pivot_y_min
    assert box.maximum.y == lever.pivot_y_max
    # the unrestricted z axis keeps a small display
    # thickness around the pivot plane
    assert box.minimum.z < lever.pivot.z < (
        box.maximum.z
    )

    # without any limits the default symmetric room
    # applies
    from dataclasses import replace

    from optimization.csv_parameter_factory import (
        DEFAULT_SHIFT_OFFSET,
    )

    stripped = MechanismDefinition(
        tuple(
            (
                replace(
                    lever,
                    pivot_x_min=None,
                    pivot_x_max=None,
                    pivot_y_min=None,
                    pivot_y_max=None,
                    pivot_z_min=None,
                    pivot_z_max=None,
                )
                for lever in inputs.definition.levers
            )
        )
    )
    box = shift_room_box(
        stripped,
        inputs.shift_lever_id,
    )
    assert box.minimum.x == (
        lever.pivot.x - DEFAULT_SHIFT_OFFSET
    )
    assert box.maximum.x == (
        lever.pivot.x + DEFAULT_SHIFT_OFFSET
    )
    assert box.minimum.y == (
        lever.pivot.y - DEFAULT_SHIFT_OFFSET
    )
    assert box.maximum.y == (
        lever.pivot.y + DEFAULT_SHIFT_OFFSET
    )


def test_stale_weight_edit_not_reapplied_after_upload_change(
    tmp_path,
):
    """Issue #141 (2): the weights of the currently
    loaded files must reach the editor and the weight
    chart.  A revised upload (weight 30 at 0 deg in the
    CSV weight column) must show up as 30 after a rerun -
    also when the same cell carried an editor edit (100)
    in the previous run."""
    pytest.importorskip("streamlit")

    from streamlit.testing.v1 import AppTest

    repo_root = Path(__file__).resolve().parents[1]

    work = tmp_path / "work"
    work.mkdir()
    (
        work / "mechanism.csv"
    ).write_text(ISSUE_MECHANISM, encoding="utf-8")
    (
        work / "target.csv"
    ).write_text(ISSUE_UNBRAKED, encoding="utf-8")
    (
        work / "target_braked.csv"
    ).write_text(ISSUE_UNBRAKED, encoding="utf-8")

    shim = tmp_path / "issue141_shim.py"
    shim.write_text(
        SHIM_TEMPLATE.replace(
            "{repo_root}",
            str(repo_root),
        ).replace(
            "{work}",
            str(work),
        ),
        encoding="utf-8",
    )

    def _bar_chart_weights() -> "list[float]":
        charts = json.loads(
            (work / "bar_charts.json").read_text()
        )
        assert charts, "no bar chart captured"
        data = charts[-1]["data"]
        return list(data["Ungebremst"])

    tester = AppTest.from_file(
        str(shim),
        default_timeout=600,
    )
    tester.run()
    assert not tester.exception

    editors = (
        list(getattr(tester, "data_editor", None) or [])
        or list(tester.dataframe)
    )
    editor = next(
        editor
        for editor in editors
        if str(editor.key).startswith(
            "two_curve_editor"
        )
    )
    assert (
        _bar_chart_weights()[5] != 100.0
    )

    tester.session_state[str(editor.key)] = {
        "edited_rows": {
            5: {"Gew. u": 100.0},
        },
        "added_rows": [],
        "deleted_rows": [],
    }
    tester.run()
    assert not tester.exception
    # the edit is applied while the upload stays the same
    assert _bar_chart_weights()[5] == 100.0

    # the user revises the file: the 0 deg weight is now 30
    (
        work / "target.csv"
    ).write_text(
        ISSUE_UNBRAKED.replace(
            "input_angle;output_angle",
            "input_angle;output_angle;weight",
        ).replace(
            "0;0",
            "0;0;30",
        ),
        encoding="utf-8",
    )
    tester.run()
    assert not tester.exception
    # the weight of the NEW file must be shown; a stale
    # cell edit of the previous run must not cover it
    weights = _bar_chart_weights()
    assert weights[5] == 30.0


SHIM_TEMPLATE = '''import sys
sys.path.insert(0, "{repo_root}")
import json
import streamlit as st
from pathlib import Path

WORK = Path("{work}")
mapping = {
    "mechanism": str(WORK / "mechanism.csv"),
    "target": str(WORK / "target.csv"),
    "target_braked": str(WORK / "target_braked.csv"),
}


class FakeUpload:
    def __init__(self, path):
        self._b = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self):
        return self._b


def fake_uploader(label, **kwargs):
    key = kwargs.get("key")
    if key in mapping:
        return FakeUpload(mapping[key])
    return None


st.file_uploader = fake_uploader

CAPTURE = []
_real_bar = st.bar_chart


def spy_bar(data=None, **kwargs):
    if isinstance(data, dict):
        CAPTURE.append(
            {
                "data": {
                    key: (
                        list(value)
                        if isinstance(
                            value, (list, tuple)
                        )
                        else value
                    )
                    for key, value in data.items()
                },
            }
        )
    return None


st.bar_chart = spy_bar

APP_FILE = str(
    Path("{repo_root}") / "gui" / "app.py",
)
_code = open(APP_FILE, encoding="utf-8").read()
exec(
    compile(_code, APP_FILE, "exec"),
    dict(__name__="__main__", __file__=APP_FILE),
)

Path(str(WORK / "bar_charts.json")).write_text(
    json.dumps(CAPTURE, default=str)
)
'''
