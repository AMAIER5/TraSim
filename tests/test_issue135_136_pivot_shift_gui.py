"""tests/test_issue135_136_pivot_shift_gui.py

Regression tests for the GitHub issues #135 and #136
(pivot shift GUI, merged target curve table):

- #136 "Optimization doesn't start": the merged target table
  passed DEGREES into ``TargetCurve.from_points_strict``,
  which expects RADIANS, so the braked curve's support points
  were at nonsense angles and evaluating it on the motion
  support raised a ValueError inside
  ``PivotShiftOptimizer.__init__``.
- #135 "Unbreaked target curve is not displayed correctly":
  the same wrong braked curve made the "Voll gebremst" chart
  series swing to thousands of degrees, squashing the
  unbraked series onto the zero line.

The tests run the real app (gui/app.py) headlessly with
Streamlit's AppTest, using the mechanism and target curve
files attached to the issues (semicolon CSV, German decimal
convention, same curve for the unbraked and the braked
state).
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = (
    Path(__file__).resolve().parents[1]
)

MECHANISM_CSV = """id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on;pivot_x_min;pivot_x_max;pivot_y_min;pivot_y_max
1;50;100;73,05799607;95;300;133,651956;0;0;239;1;0;0;;;;;;;;;;
2;50;100;76,9929424;100;260;153,1432978;0;0;-261;1;0;0;;;;1;;;-50;50;-50;50
3;50;100;91,73216337;-90;90;11,8383644;0;0;-261;1;0;0;;;;;2;;;;;
4;50;100;62,09694189;-70;70;-5,795825305;0;-40;-1020;1;0;0;;;;3;;;;;;
5;50;100;92,07312009;60;300;147,4731758;-815;-40;-1020;1;0;0;;;;;4;;;;;;
6;70;85;82;30;130;80;-850;-218;-4020;0;1;0;;;;5;;;;;;
"""

TARGET_CURVE_CSV = """input_angle;output_angle
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

SHIM_TEMPLATE = '''
import sys
sys.path.insert(0, "{repo_root}")
import json
import streamlit as st
from pathlib import Path

WORK = Path("{work}")
mapping = {
    "mechanism": str(WORK / "mechanism_initial.csv"),
    "target": str(WORK / "target_curve_unbreaked.csv"),
    "target_braked": str(WORK / "target_curve_unbreaked.csv"),
}


class FakeUpload:
    def __init__(self, path):
        self._b = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self):
        return self._b


def fake_uploader(label, **kwargs):
    key = kwargs.get("key")
    st.file_uploader = fake_uploader
    if key in mapping:
        return FakeUpload(mapping[key])
    return None


st.file_uploader = fake_uploader

CAPTURE = []
_real_line = st.line_chart


def spy_line(data=None, **kwargs):
    if isinstance(data, dict):
        CAPTURE.append(
            (
                "DICT",
                {
                    key: (
                        list(value)
                        if isinstance(
                            value,
                            (list, tuple),
                        )
                        else value
                    )
                    for key, value in (
                        data.items()
                    )
                },
                kwargs.get("y"),
            )
        )
    elif isinstance(data, list):
        CAPTURE.append(("ROWS", data, kwargs.get("y")))
    return _real_line(data, **kwargs)


st.line_chart = spy_line

APP_FILE = str(
    Path("{repo_root}") / "gui" / "app.py",
)
_code = open(APP_FILE, encoding="utf-8").read()
exec(
    compile(_code, APP_FILE, "exec"),
    dict(__name__="__main__", __file__=APP_FILE),
)

Path(str(WORK / "charts.json")).write_text(
    json.dumps(
        [
            {
                "kind": kind,
                "data": data,
                "y": y,
            }
            for kind, data, y in CAPTURE
        ],
        default=str,
    )
)
'''


def _prepare_workdir(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    (work / "mechanism_initial.csv").write_text(
        MECHANISM_CSV,
        encoding="utf-8",
    )
    (work / "target_curve_unbreaked.csv").write_text(
        TARGET_CURVE_CSV,
        encoding="utf-8",
    )
    return work


def _write_shim(tmp_path: Path, work: Path) -> Path:
    shim = tmp_path / "issue_shim.py"
    shim.write_text(
        SHIM_TEMPLATE.replace(
            "{repo_root}",
            str(REPO_ROOT),
        ).replace(
            "{work}",
            str(work),
        ),
        encoding="utf-8",
    )
    return shim


def _set_sidebar_values(tester):
    pairs = (
        ("Populationsgr\u00f6\u00dfe", 10),
        ("Maximale Generationen", 10),
        ("Kinder pro Generation", 6),
        ("Stagnationslimit (Generationen)", 2),
        ("Worker-Prozesse", 1),
    )
    for label, value in pairs:
        for number_input in (
            tester.sidebar.number_input
        ):
            if label == (
                number_input.label or ""
            ):
                number_input.set_value(value)
                break


def test_issue135_both_curves_in_degree_range(
    tmp_path,
):
    """The unbraked and the braked target curve of the
    shared table must both display in the same, plausible
    degree range (not thousands of degrees, not the
    unbraked line squashed onto zero)."""
    work = _prepare_workdir(tmp_path)
    shim = _write_shim(tmp_path, work)
    tester = AppTest.from_file(
        str(shim),
        default_timeout=600,
    )
    tester.run()
    assert not tester.exception
    charts = json.loads(
        (work / "charts.json").read_text()
    )
    curve_chart = next(
        chart
        for chart in charts
        if chart["kind"] == "DICT"
        and any(
            key.startswith("Ungebremst")
            for key in chart["data"]
        )
    )
    unbraked = curve_chart["data"][
        "Ungebremst (°)"
    ]
    braked = curve_chart["data"][
        "Voll gebremst (°)"
    ]
    assert all(
        abs(value) <= 90.0
        for value in unbraked
    )
    assert all(
        abs(value) <= 90.0
        for value in braked
    )
    # Issue #135: the unbraked series was visually constant
    # at 0 because of the braked series' absurd scale; a real
    # curve must have a non-vanishing span.
    assert (
        max(unbraked) - min(unbraked)
    ) > 10.0


def test_issue136_optimization_starts(tmp_path):
    """Clicking "Optimierung starten" in the pivot shift
    mode must complete the run without the ValueError from
    the wrongly converted braked target curve."""
    work = _prepare_workdir(tmp_path)
    shim = _write_shim(tmp_path, work)
    tester = AppTest.from_file(
        str(shim),
        default_timeout=900,
    )
    tester.run()
    assert not tester.exception
    _set_sidebar_values(tester)
    start = next(
        button
        for button in tester.button
        if button.label
        == "Optimierung starten"
    )
    start.click()
    tester.run()
    assert not tester.exception
    result_text = next(
        (
            element.value
            for element in tester.markdown
            if "Stopgrund" in (
                element.value or ""
            )
        ),
        None,
    )
    assert result_text is not None
    # With the initial angular offset between the virtual
    # drive shaft and the start lever (drive angle 133.65
    # deg, mean of the prescribed input values 0 deg), the
    # drive lever sweeps inside its admissible lever-angle
    # segment [95, 300] deg: the run must score a real
    # fitness instead of the blocking hard penalty.
    marker = "Beste Fitness:** "
    tail = result_text.split(marker, 1)[1]
    value = float(tail.split()[0].strip())
    assert value < 90000.0
