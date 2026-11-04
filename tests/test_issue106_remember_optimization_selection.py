"""
tests/test_issue106_remember_optimization_selection.py

Issue #106: the selection of the optimization-released
parameters (column "Optimierung" in the table "Parameter des
Startmechanismus") must only be reset to the initial state when
a new mechanism is loaded.  In all other cases (restarting the
optimization with changed parameters, editing single parameters,
adopting optimized values) the selection set by the user must
remain.

The GUI is exercised with streamlit's AppTest so the session
state handling of gui/app.py is covered directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).parent.parent
APP_FILE = REPO_ROOT / "gui" / "app.py"
EXAMPLES = REPO_ROOT / "examples"


def _load_app() -> AppTest:
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
    return app


def _mechanism_editor(app: AppTest):
    editors = (
        list(getattr(app, "data_editor", None) or [])
        or list(app.dataframe)
    )
    return next(
        editor
        for editor in editors
        if str(editor.key).startswith("mechanism_editor")
    )


def _uncheck_length_optimize(
    app: AppTest,
    row_index: int = 0,
) -> None:
    editor = _mechanism_editor(app)
    app.session_state[str(editor.key)] = {
        "edited_rows": {row_index: {"length_optimize": False}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert not app.exception


def _length_optimize_flags(app: AppTest) -> list[bool]:
    editor = _mechanism_editor(app)
    return [
        bool(value) for value in editor.value["length_optimize"]
    ]


def test_selection_survives_reruns():
    app = _load_app()
    _uncheck_length_optimize(app)
    assert (
        app.session_state["mechanism_rows"][0][
            "length_optimize"
        ]
        is False
    )
    app.sidebar.number_input[0].set_value(
        app.sidebar.number_input[0].value + 1
    )
    app.run()
    assert not app.exception
    assert _length_optimize_flags(app)[0] is False


def test_selection_reset_only_on_new_mechanism():
    app = _load_app()
    _uncheck_length_optimize(app)
    app.file_uploader[0].set_value(
        (
            "other_mechanism.csv",
            (EXAMPLES / "PPBLS_mechanism.csv").read_bytes(),
            "text/csv",
        )
    )
    app.run()
    assert not app.exception
    assert all(_length_optimize_flags(app))
