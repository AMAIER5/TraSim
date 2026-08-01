"""
solver/angle_solver.py

Local numerical angle solver using Brent root finding.

The solver preserves the physical motion branch by searching
around the previous solution stored in SolverState.
"""

from __future__ import annotations

import math

from mechanics.stage import Stage
from solver.objective import stage_error
from solver.root_solver import RootSolver
from solver.solver_result import SolverResult
from solver.solver_state import SolverState


class AngleSolver:
    """
    Solve output angle for a given input angle.

    A local bracket is searched around the previous output angle.
    The resulting interval is solved using Brent's method.
    """

    def __init__(
        self,
        *,
        search_window: float = math.radians(45),
        bracket_step: float = math.radians(1),
        tolerance: float = 1e-10,
        max_iterations: int = 40,
    ):
        self.search_window = search_window
        self.bracket_step = bracket_step
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(
        self,
        stage: Stage,
        input_angle: float,
        state: SolverState,
    ) -> SolverResult:
        """
        Find output angle for input angle.
        """

        def residual(angle: float) -> float:
            return stage_error(
                stage,
                input_angle,
                angle,
            )

# DEBUG CODE BEGIN
        for deg in range(-30, 31, 5):
            angle = math.radians(deg)
            print(
                deg,
                stage_error(
                    stage,
                    input_angle,
                    angle,
                ),
            )
# DEBUG CODE END

        bracket = RootSolver.find_bracket(
            function=residual,
            center=state.last_output_angle,
            window=self.search_window,
            step=self.bracket_step,
        )

        if bracket is None:

            return SolverResult(
                success=False,
                angle=float("nan"),
                residual=float("inf"),
                iterations=0,
                reason="blocked",
            )

        left, right, bracket_iterations = bracket

        #
        # Exact solution already found
        #

        if left == right:

            value = residual(left)

            return SolverResult(
                success=abs(value) <= self.tolerance,
                angle=left,
                residual=abs(value),
                iterations=bracket_iterations,
                reason=None,
            )

        try:

            angle, value, solver_iterations = (
                RootSolver.solve_brent(
                    function=residual,
                    left=left,
                    right=right,
                    tolerance=self.tolerance,
                    max_iterations=self.max_iterations,
                )
            )

        except ValueError:

            return SolverResult(
                success=False,
                angle=float("nan"),
                residual=float("inf"),
                iterations=bracket_iterations,
                reason="invalid_bracket",
            )

        iterations = (
            bracket_iterations
            +
            solver_iterations
        )

        success = (
            abs(value)
            <= self.tolerance
        )

        return SolverResult(
            success=success,
            angle=angle if success else float("nan"),
            residual=abs(value),
            iterations=iterations,
            reason=None if success else "blocked",
        )