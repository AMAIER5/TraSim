"""run_trasim.py

Launcher for the packaged Streamlit app.

When the app is frozen with PyInstaller
(--onefile), this module is the entry point that
starts Streamlit programmatically on the user's
own machine.

PyInstaller bundles the ``gui`` package because
``gui.workflow`` is imported here directly.  The
app script itself is included as data
(--add-data) and located at runtime via
``sys._MEIPASS`` when frozen.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gui import workflow  # noqa: F401


def _app_path() -> str:
    meipass = getattr(
        sys,
        "_MEIPASS",
        None,
    )
    if meipass is not None:
        return str(
            Path(meipass) / "gui" / "app.py",
        )
    return str(
        Path(__file__).resolve().parent
        / "gui"
        / "app.py",
    )


def main() -> int:
    sys.argv = [
        "streamlit",
        "run",
        _app_path(),
        "--global.developmentMode=false",
    ]
    from streamlit.web import cli as stcli

    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
