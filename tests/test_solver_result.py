"""
tests/test_solver_result.py

Unit tests for SolverResult.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from solver.solver_result import SolverResult


# ---------------------------------------------------------------------------
# Successful result
# ---------------------------------------------------------------------------

def test_solver_result_success():

    result = SolverResult(
        success=True,
        angle=1.234,
        residual=1e-10,
        iterations=5,
    )

    assert result.success is True
    assert math.isclose(result.angle, 1.234)
    assert math.isclose(result.residual, 1e-10)
    assert result.iterations == 5
    assert result.reason is None


# ---------------------------------------------------------------------------
# Failed result
# ---------------------------------------------------------------------------

def test_solver_result_failure():

    result = SolverResult(
        success=False,
        angle=float("nan"),
        residual=float("inf"),
        iterations=20,
        reason="blocked",
    )

    assert result.success is False
    assert math.isnan(result.angle)
    assert math.isinf(result.residual)
    assert result.iterations == 20
    assert result.reason == "blocked"


# ---------------------------------------------------------------------------
# Immutable result
# ---------------------------------------------------------------------------

def test_solver_result_is_immutable():

    result = SolverResult(
        success=True,
        angle=0.0,
        residual=0.0,
        iterations=0,
    )

    with pytest.raises(FrozenInstanceError):

        result.success = False


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

def test_solver_result_equality():

    result_a = SolverResult(
        success=True,
        angle=1.0,
        residual=0.0,
        iterations=3,
    )

    result_b = SolverResult(
        success=True,
        angle=1.0,
        residual=0.0,
        iterations=3,
    )

    assert result_a == result_b