"""
tests/debug_simulation.py

Issue #16: This file was a diagnostic script masquerading
as a test — ``test_debug_simulation`` contained only
``print()`` statements and no assertions.  It would
always be picked up by pytest and, depending on
fixtures, could fail or produce confusing output.

The diagnostic logic has been moved to
``examples/debug_simulation.py`` (function renamed to
``run_debug_simulation``).

This file now keeps the test name for backward
compatibility but marks it ``@pytest.mark.skip`` with a
clear reason, redirecting users to the examples module.
"""

import pytest

from examples.debug_simulation import run_debug_simulation


@pytest.mark.skip(
    reason=(
        "Issue #16: This diagnostic script has been moved "
        "to examples/debug_simulation.py.  Run it directly "
        "or use: python -m examples.debug_simulation <csv>"
    ),
)
def test_debug_simulation(
    simple_stage_csv,
):
    """
    Diagnostic script — skipped.

    See ``examples/debug_simulation.py`` for the active
    version of this debugging aid.
    """

    run_debug_simulation(simple_stage_csv)