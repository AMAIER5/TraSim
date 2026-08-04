"""
solver/solver_precision.py

Numerical precision configuration for kinematic solvers.

This module contains solver-specific numerical parameters.
It is intentionally independent from optimization and fitness logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SolverPrecision:
    """
    Numerical configuration for solver accuracy.

    The values define the trade-off between:
    - simulation speed
    - numerical accuracy
    - solver robustness

    Angles are stored in radians.
    """

    tolerance: float = 1e-10

    max_iterations: int = 40

    bracket_step: float = math.radians(1)

    search_window: float = math.radians(30)

    def __post_init__(self) -> None:

        if self.tolerance <= 0.0:
            raise ValueError(
                "tolerance must be positive."
            )

        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be greater than zero."
            )

        if self.bracket_step <= 0.0:
            raise ValueError(
                "bracket_step must be positive."
            )

        if self.search_window <= 0.0:
            raise ValueError(
                "search_window must be positive."
            )