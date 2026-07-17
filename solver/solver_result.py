"""
solver/solver_result.py

Common result container for numerical solvers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverResult:
    """
    Result returned by a solver.

    Parameters
    ----------
    success:
        True if a valid solution was found.

    angle:
        Calculated angle [rad].

        If no solution exists:
        NaN is returned.

    residual:
        Remaining constraint error.

    iterations:
        Number of solver iterations.

    reason:
        Optional explanation for failed solutions.
    """

    success: bool

    angle: float

    residual: float

    iterations: int

    reason: str | None = None