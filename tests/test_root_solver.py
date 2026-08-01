"""
tests/test_root_solver.py

Tests for the generic numerical root solver.
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