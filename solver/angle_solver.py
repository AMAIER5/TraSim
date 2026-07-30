"""
solver/angle_solver.py

Local search based angle solver with refinement.
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

    A coarse search is followed by a local refinement step
    to improve angular accuracy.
    """

    def __init__(
        self,
        *,
        search_window: float = math.radians(45),
        search_step: float = math.radians(0.5),
        refinement_steps: int = 4,
        tolerance: float = 1e-3,
    ):
        self.search_window = search_window
        self.search_step = search_step
        self.refinement_steps = refinement_steps
        self.tolerance = tolerance

    def solve(
        self,
        stage: Stage,
        input_angle: float,
        state: SolverState,
    ) -> SolverResult:
        """
        Find output angle for input angle.
        """

        start = (
            state.last_output_angle
            -
            self.search_window
        )

        end = (
            state.last_output_angle
            +
            self.search_window
        )

        best_angle = float("nan")
        best_error = float("inf")

        iterations = 0

        # -------------------------------------------------
        # Coarse search
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Local refinement
        # -------------------------------------------------

        refinement_step = self.search_step

        for _ in range(
            self.refinement_steps
        ):

            refinement_step *= 0.1

            start = -math.pi
            end = math.pi

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

                angle += refinement_step


        # -------------------------------------------------
        # Result
        # -------------------------------------------------

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