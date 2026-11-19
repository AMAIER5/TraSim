"""Tests for issue #126: the first input in the GUI
parameter table "Parameter des Startmechanismus" was
often withdrawn and reset to the previous value.

The mechanism table's data editor now persists its edit
diff the moment the edit event is processed (via an
``on_change`` callback and ``merge_editor_edits``), so
the first input is durable even when the rerun triggered
by the edit aborts early.  The session-state flow is
exercised with streamlit's AppTest like in the existing
GUI tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from gui.workflow import merge_editor_edits

REPO_ROOT = Path(__file__).parent.parent
APP_FILE = REPO_ROOT / "gui" / "app.py"
EXAMPLES = REPO_ROOT / "examples"


def test_merge_editor_edits_applies_edited_rows():
    rows = [
        {"id": 1, "length": 50.0, "angle": 0.0},
        {"id": 2, "length": 60.0, "angle": 10.0},
    ]
    merged = merge_editor_edits(
        rows,
        {
            "edited_rows": {0: {"length": 55.0}},
            "added_rows": [],
            "deleted_rows": [],
        },
    )
    assert merged[0]["length"] == 55.0
    assert merged[1]["length"] == 60.0
    # the source rows stay untouched
    assert rows[0]["length"] == 50.0


def test_merge_editor_edits_handles_add_and_delete():
    rows = [
        {"id": 1, "length": 50.0},
        {"id": 2, "length": 60.0},
    ]
    merged = merge_editor_edits(
        rows,
        {
            "edited_rows": {},
            "added_rows": [
                {"id": 3, "length": 70.0},
            ],
            "deleted_rows": [1],
        },
    )
    assert [
        row["id"] for row in merged
    ] == [1, 3]


def test_merge_editor_edits_tolerates_missing_state():
    rows = [{"id": 1, "length": 50.0}]
    assert merge_editor_edits(rows, None) == [
        {"id": 1, "length": 50.0},
    ]
    assert (
        merge_editor_edits(
            rows,
            {
                "edited_rows": None,
                "added_rows": None,
                "deleted_rows": None,
            },
        )
        == rows
    )


def _load_app() -> AppTest:
    app = AppTest.from_file(str(APP_FILE), default_timeout=120)
    app.run()
    app.file_uploader[0].set_value(
        (
            "mechanism.csv",
            (EXAMPLES / "mechanism.csv").read_bytes(),
            "text/csv",
        ),
    )
    app.file_uploader[1].set_value(
        (
            "targetcurve.csv",
            (EXAMPLES / "target_curve.csv").read_bytes(),
            "text/csv",
        ),
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
        if str(editor.key).startswith(
            "mechanism_editor",
        )
    )


def test_first_edit_survives_reruns() -> None:
    app = _load_app()
    editor = _mechanism_editor(app)
    old_length = app.session_state[
        "mechanism_rows"
    ][0]["length"]
    new_length = old_length + 5.0
    app.session_state[str(editor.key)] = {
        "edited_rows": {
            0: {"length": new_length},
        },
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert not app.exception
    assert (
        app.session_state["mechanism_rows"][0][
            "length"
        ]
        == new_length
    )
    # a plain rerun (any other widget interaction) must
    # not withdraw the first input either
    app.run()
    assert not app.exception
    assert (
        app.session_state["mechanism_rows"][0][
            "length"
        ]
        == new_length
    )


def test_edited_rows_source_merged_by_callback_helper() -> None:
    # Issue #126: the on_change callback merges the
    # widget diff into the source rows before the script
    # runs; verify the merged result is what the script
    # then feeds back into the table.
    app = _load_app()
    editor = _mechanism_editor(app)
    rows = app.session_state["mechanism_rows"]
    edits = {
        "edited_rows": {
            0: {"length": rows[0]["length"] + 1.0},
        },
        "added_rows": [],
        "deleted_rows": [],
    }
    merged = merge_editor_edits(rows, edits)
    app.session_state[str(editor.key)] = edits
    app.session_state["mechanism_rows"] = merged
    app.run()
    assert not app.exception
    assert (
        app.session_state["mechanism_rows"][0][
            "length"
        ]
        == merged[0]["length"]
    )
