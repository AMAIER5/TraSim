"""tests/test_issue100_gui_chart_columns.py

Regression test for issue #100
("GUI: Optimization without mixing can't be loaded any
more"): uploading a mechanism CSV without a ``pivot_on``
column (a normal, non-mixed optimization) crashed the GUI
with

    StreamlitAPIException: Data does not have a column
    named "Voll gebremst (*)"

because ``st.line_chart``/``st.bar_chart`` were called with
a hardcoded y-column list that always contains the mixed
brake column ``Voll gebremst``, even though that column is
only added to the chart data in mixed mode (pivot_on).

``gui/app.py`` is a top-level Streamlit script and cannot be
imported in tests, so this test inspects its source: the
``y`` argument of the chart calls must be a computed list
(not a hardcoded list literal), and the ``Voll gebremst``
column may only be appended conditionally.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP_FILE = (
    Path(__file__).parent.parent
    / "gui"
    / "app.py"
)

BRAKED_COLUMN = "Voll gebremst"
BRAKED_COLUMN_DEG = "Voll gebremst (°)"


def _app_tree() -> ast.Module:
    return ast.parse(
        APP_FILE.read_text(encoding="utf-8")
    )


def _chart_calls(
    tree: ast.Module,
) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "st"
            and func.attr
            in ("line_chart", "bar_chart")
        ):
            calls.append(node)
    assert calls, (
        "expected st.line_chart/st.bar_chart calls "
        "in gui/app.py"
    )
    return calls


def test_chart_y_columns_are_computed():
    """The y argument of every chart call must be a
    computed name (a variable), not a hardcoded list
    literal containing the braked column (issue #100)."""
    for call in _chart_calls(_app_tree()):
        y = None
        for keyword in call.keywords:
            if keyword.arg == "y":
                y = keyword.value
                break
        if y is None:
            continue
        assert isinstance(y, ast.Name), (
            "the chart y columns must be computed from "
            "the chart data, not hardcoded as a list "
            "literal (issue #100)"
        )


def test_braked_column_appended_conditionally():
    """The braked y column may only be added to the
    computed column list when it exists in the chart
    data (mixed mode)."""
    source = APP_FILE.read_text(encoding="utf-8")
    assert (
        f'"{BRAKED_COLUMN_DEG}" in chart_data'
        in source.replace("'", '"')
    )
    assert f'"{BRAKED_COLUMN}" in weight_data' in (
        source.replace("'", '"')
    )


def test_normal_mechanism_without_pivot_on_loads():
    """End-to-end sanity check that a mechanism CSV
    without a pivot_on column loads through the GUI
    workflow used by app.py."""
    from gui.workflow import (
        WorkflowError,
        load_inputs,
    )

    examples = (
        Path(__file__).parent.parent / "examples"
    )
    try:
        inputs = load_inputs(
            (
                examples / "levers.csv"
            ).read_text(encoding="utf-8"),
            (
                examples / "target_curve.csv"
            ).read_text(encoding="utf-8"),
        )
    except WorkflowError as error:
        raise AssertionError(
            f"normal mechanism must load: {error}"
        )
    assert inputs.point_count > 0
    assert all(
        lever.pivot_on is None
        for lever in inputs.definition.levers
    )
