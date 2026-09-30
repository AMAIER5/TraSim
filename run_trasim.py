"""run_trasim.py

Launcher for the packaged Streamlit app.

When the app is frozen with PyInstaller
(--onefile), this module is the entry point that
starts Streamlit programmatically on the user's
own machine.
"""

from __future__ import annotations

import sys

from streamlit.web import cli as stcli


def main() -> int:
    sys.argv = [
        "streamlit",
        "run",
        "gui/app.py",
        "--global.developmentMode=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
