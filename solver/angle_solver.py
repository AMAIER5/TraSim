"""
solver/angle_solver.py

Local search based angle solver.
"""

from __future__ import annotations

import math

from mechanics.stage import Stage
from solver.objective import stage_error
from solver.solver_result import SolverResult
from solver.solver_state import SolverState


class AngleSolver:
    """
    Solve output angle for a given input angle.

    The solver searches around the previously known solution
    to preserve the physical motion branch.
    """

    def __init__(
        self,
        *,
        search_window: float = math.radians(15),
        search_step: float = math.radians(0.25),
        tolerance: float = 1e-9,
    ):
        self.search_window = search_window
        self.search_step = search_step
        self.tolerance = tolerance

    def solve(
        self,
        stage: Stage,
        input_angle: float,
        state: SolverState,
    ) -> SolverResult:
        """
        Find output angle for input angle.

        Parameters
        ----------
        stage:
            Mechanical stage.

        input_angle:
            Input rotation angle [rad].

        state:
            Previous solver state.
        """

        start = (
            state.last_output_angle
            - self.search_window
        )

        end = (
            state.last_output_angle
            + self.search_window
        )

        best_angle = float("nan")
        best_error = float("inf")

        iterations = 0

        angle = start

        while angle <= end:

            error = abs(
                stage_error(
                    stage,
                    input_angle,
                    angle,
                )
            )

            iterations += 1

            if error < best_error:
                best_error = error
                best_angle = angle

            angle += self.search_step

        if best_error <= self.tolerance:

            return SolverResult(
                success=True,
                angle=best_angle,
                residual=best_error,
                iterations=iterations,
            )

        return SolverResult(
            success=False,
            angle=float("nan"),
            residual=best_error,
            iterations=iterations,
            reason="blocked",
        )