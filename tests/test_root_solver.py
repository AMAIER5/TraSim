"""
tests/test_root_solver.py

Tests for the generic numerical root solver.

Issue #20: Added tests for Brent iteration counting
(starts at 0, counts refinement steps) and convergence
on edge cases (tight tolerance, near-zero root,
asymmetric brackets).
"""

from __future__ import annotations

import math

import pytest

from solver.root_solver import RootSolver


def test_find_bracket_positive_root():
    """
    The bracket search finds a sign change around the center.
    """

    def function(x: float) -> float:
        return x - 2.0

    result = RootSolver.find_bracket(
        function=function,
        center=0.0,
        window=5.0,
        step=0.5,
    )

    assert result is not None

    left, right, evaluations = result

    assert left <= 2.0
    assert right >= 2.0
    assert evaluations > 0


def test_find_bracket_negative_direction():
    """
    The bracket search also works in the negative direction.
    """

    def function(x: float) -> float:
        return x + 3.0

    result = RootSolver.find_bracket(
        function=function,
        center=0.0,
        window=5.0,
        step=0.5,
    )

    assert result is not None

    left, right, _ = result

    assert left <= -3.0
    assert right >= -3.0


def test_find_bracket_no_solution():
    """
    A function without real roots does not produce a bracket.
    """

    def function(x: float) -> float:
        return x * x + 1.0

    result = RootSolver.find_bracket(
        function=function,
        center=0.0,
        window=5.0,
        step=0.5,
    )

    assert result is None


def test_brent_solves_quadratic_root():
    """
    Brent converges to a known quadratic root.
    """

    def function(x: float) -> float:
        return x * x - 4.0

    root, residual, iterations = RootSolver.solve_brent(
        function=function,
        left=0.0,
        right=5.0,
    )

    assert root == pytest.approx(
        2.0,
        abs=1e-10,
    )

    assert residual == pytest.approx(
        0.0,
        abs=1e-10,
    )

    assert iterations < 40


def test_brent_solves_negative_root():
    """
    Brent handles negative roots.
    """

    def function(x: float) -> float:
        return x + 3.0

    root, residual, _ = RootSolver.solve_brent(
        function=function,
        left=-5.0,
        right=0.0,
    )

    assert root == pytest.approx(
        -3.0,
        abs=1e-10,
    )

    assert residual == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_brent_requires_bracket():
    """
    Brent rejects intervals without a sign change.
    """

    def function(x: float) -> float:
        return x * x + 1.0

    with pytest.raises(ValueError):

        RootSolver.solve_brent(
            function=function,
            left=-1.0,
            right=1.0,
        )


def test_brent_iteration_limit():
    """
    Brent remains within the configured iteration limit.
    """

    def function(x: float) -> float:
        return math.cos(x) - x

    _, _, iterations = RootSolver.solve_brent(
        function=function,
        left=0.0,
        right=1.0,
        max_iterations=40,
    )

    assert iterations <= 40


# ---------------------------------------------------------------------------
# Issue #20: Iteration counting and convergence edge cases
# ---------------------------------------------------------------------------

def test_brent_iterations_start_at_zero_for_exact_root():
    """
    Issue #20: When fa == 0.0 at the left boundary,
    iterations must be 0 (not 2 as in the old code).
    """

    def function(x: float) -> float:
        return x - 5.0

    root, residual, iterations = RootSolver.solve_brent(
        function=function,
        left=5.0,
        right=10.0,
    )

    assert root == pytest.approx(5.0)
    assert iterations == 0


def test_brent_iterations_start_at_zero_for_exact_right_root():
    """
    Issue #20: When fb == 0.0 at the right boundary,
    iterations must be 0.
    """

    def function(x: float) -> float:
        return x - 5.0

    root, residual, iterations = RootSolver.solve_brent(
        function=function,
        left=0.0,
        right=5.0,
    )

    assert root == pytest.approx(5.0)
    assert iterations == 0


def test_brent_iteration_count_is_refinement_steps():
    """
    Issue #20: iterations counts refinement steps,
    not function evaluations.  It should be a small
    number for a well-behaved function.
    """

    def function(x: float) -> float:
        return x - 1.0

    # Linear function: Brent should converge very fast.
    _, _, iterations = RootSolver.solve_brent(
        function=function,
        left=0.0,
        right=2.0,
    )

    # A linear function converges in very few steps.
    assert iterations < 10


def test_brent_tight_tolerance():
    """
    Issue #20: Brent must converge to a tight tolerance.
    """

    def function(x: float) -> float:
        return x * x - 2.0

    root, residual, iterations = RootSolver.solve_brent(
        function=function,
        left=1.0,
        right=2.0,
        tolerance=1e-15,
    )

    assert root == pytest.approx(
        math.sqrt(2.0),
        abs=1e-14,
    )

    assert iterations <= 40


def test_brent_near_zero_root():
    """
    Issue #20: Brent handles a root very close to zero
    where machine-epsilon guards matter.
    """

    def function(x: float) -> float:
        return x - 1e-10

    root, residual, _ = RootSolver.solve_brent(
        function=function,
        left=-1.0,
        right=1.0,
        tolerance=1e-15,
    )

    assert root == pytest.approx(1e-10, abs=1e-15)


def test_brent_asymmetric_bracket():
    """
    Issue #20: Brent converges even with a very
    asymmetric initial bracket.
    """

    def function(x: float) -> float:
        return x - 3.0

    root, residual, _ = RootSolver.solve_brent(
        function=function,
        left=-1000.0,
        right=3.0001,
    )

    assert root == pytest.approx(3.0, abs=1e-10)


def test_brent_large_root():
    """
    Issue #20: Brent handles a large root where the
    machine-epsilon term dominates the tolerance.
    """

    def function(x: float) -> float:
        return x - 1e12

    root, residual, _ = RootSolver.solve_brent(
        function=function,
        left=0.0,
        right=2e12,
    )

    assert root == pytest.approx(1e12, rel=1e-12)


def test_brent_cubic_root():
    """
    Issue #20: Brent converges on a cubic function.
    """

    def function(x: float) -> float:
        return x ** 3 - 27.0

    root, residual, iterations = RootSolver.solve_brent(
        function=function,
        left=2.0,
        right=4.0,
    )

    assert root == pytest.approx(3.0, abs=1e-10)
    assert iterations <= 40


def test_brent_tangent_function():
    """
    Issue #20: A function that is tangent to zero
    (no sign change but touches) — Brent should still
    work if the bracket has a sign change.
    """

    def function(x: float) -> float:
        # Root at x=1, crosses sign.
        return (x - 1.0) ** 3

    root, residual, _ = RootSolver.solve_brent(
        function=function,
        left=0.0,
        right=2.0,
    )

    assert root == pytest.approx(1.0, abs=1e-10)