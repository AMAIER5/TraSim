"""
solver/angle_solver.py

Local numerical angle solver using Brent root finding.

The solver preserves the physical motion branch by searching
all possible roots around the predicted motion state and
selecting the physically continuous branch.

Performance optimization:
- adaptive local bracket reuse
- dynamic search window based on previous root movement
- single bracket fast path

Branch selection considers:
- prediction distance
- output angle jump
- velocity change
- motion direction
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

    Multiple mathematical solutions may exist.

    The solver keeps the physical motion branch by using
    SolverState prediction and local kinematic continuity.
    """

    def __init__(
        self,
        *,
        search_min: float = math.radians(-180),
        search_max: float = math.radians(180),
        bracket_step: float = math.radians(1),
        search_window: float = math.radians(30),
        reuse_factor: float = 3.0,
        reuse_min_window: float = math.radians(2),
        reuse_max_window: float = math.radians(15),
        tolerance: float = 1e-10,
        max_iterations: int = 40,
    ) -> None:

        self.search_min = search_min
        self.search_max = search_max
        self.bracket_step = bracket_step
        self.search_window = search_window

        self.reuse_factor = reuse_factor
        self.reuse_min_window = reuse_min_window
        self.reuse_max_window = reuse_max_window

        self.tolerance = tolerance
        self.max_iterations = max_iterations

        #
        # Adaptive bracket cache
        #

        self._last_root: float | None = None
        self._last_bracket_width = bracket_step

        #
        # Performance statistics
        #

        self.stats = {
            "adaptive_attempts": 0,
            "adaptive_success": 0,
            "adaptive_failure": 0,
            "local_searches": 0,
            "fallback_searches": 0,
            "brackets_found": 0,
            "single_bracket_fast_path": 0,
            "multi_bracket_selection": 0,
            "solved": 0,
            "blocked": 0,
        }


    def solve(
        self,
        stage: Stage,
        input_angle: float,
        state: SolverState,
    ) -> SolverResult:

        def residual(angle: float) -> float:
            return stage_error(
                stage,
                input_angle,
                angle,
            )


        predicted = state.predict_output(
            input_angle
        )


        if predicted is None:
            predicted = state.last_output_angle


        #
        # Adaptive local bracket reuse
        #

        brackets = []

        if self._last_root is not None:

            self.stats["adaptive_attempts"] += 1

            window = min(
                max(
                    self._last_bracket_width
                    *
                    self.reuse_factor,
                    self.reuse_min_window,
                ),
                self.reuse_max_window,
            )


            brackets = RootSolver.find_all_brackets_around(
                function=residual,
                center=predicted,
                window=window,
                step=self.bracket_step,
            )


            if brackets:
                self.stats["adaptive_success"] += 1
            else:
                self.stats["adaptive_failure"] += 1


        #
        # Normal local search
        #

        if not brackets:

            self.stats["local_searches"] += 1

            brackets = RootSolver.find_all_brackets_around(
                function=residual,
                center=predicted,
                window=self.search_window,
                step=self.bracket_step,
            )


        #
        # Complete search fallback
        #

        if not brackets:

            self.stats["fallback_searches"] += 1

            brackets = RootSolver.find_all_brackets(
                function=residual,
                minimum=self.search_min,
                maximum=self.search_max,
                step=self.bracket_step,
            )


        self.stats["brackets_found"] += len(brackets)


        if not brackets:

            self.stats["blocked"] += 1

            return SolverResult(
                success=False,
                angle=float("nan"),
                residual=float("inf"),
                iterations=0,
                reason="blocked",
            )


        #
        # Single bracket fast path
        #
        # A single mathematical root has no ambiguity.
        # Avoid expensive physical branch scoring.
        #

        if len(brackets) == 1:

            self.stats["single_bracket_fast_path"] += 1

            bracket = brackets[0]

        else:

            self.stats["multi_bracket_selection"] += 1

            bracket = self._select_branch(
                brackets,
                reference_angle=predicted,
                state=state,
                input_angle=input_angle,
            )


        left, right, bracket_iterations = bracket


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

            self.stats["blocked"] += 1

            return SolverResult(
                success=False,
                angle=float("nan"),
                residual=float("inf"),
                iterations=bracket_iterations,
                reason="invalid_bracket",
            )


        success = (
            abs(value)
            <= self.tolerance
        )


        if success:

            self.stats["solved"] += 1

            #
            # Update adaptive cache
            #

            self._last_root = angle

            self._last_bracket_width = abs(
                right - left
            )

        else:

            self.stats["blocked"] += 1


        return SolverResult(
            success=success,
            angle=angle if success else float("nan"),
            residual=abs(value),
            iterations=(
                bracket_iterations
                +
                solver_iterations
            ),
            reason=None if success else "blocked",
        )


    def get_stats(self) -> dict[str, int]:

        """
        Return solver performance statistics.
        """

        return self.stats.copy()


    def reset_stats(self) -> None:

        """
        Reset solver performance statistics.
        """

        for key in self.stats:
            self.stats[key] = 0


        self._last_root = None
        self._last_bracket_width = self.bracket_step


    @staticmethod
    def _select_branch(
        brackets: list[tuple[float, float, int]],
        reference_angle: float,
        state: SolverState,
        input_angle: float,
    ) -> tuple[float, float, int]:

        """
        Select the physically continuous branch.
        """

        if state.direction not in (-1, 0, 1):

            raise ValueError(
                "direction must be either -1, 0 or 1"
            )


        def center(
            bracket: tuple[float, float, int],
        ) -> float:

            left, right, _ = bracket

            return (
                left + right
            ) / 2.0


        def score(
            bracket: tuple[float, float, int],
        ) -> float:

            candidate = center(bracket)


            prediction_error = abs(
                candidate
                -
                reference_angle
            )


            if state.direction == 0:

                return prediction_error


            output_change = abs(
                candidate
                -
                state.last_output_angle
            )


            delta_input = (
                input_angle
                -
                state.last_input_angle
            )


            if abs(delta_input) > 1e-12:

                velocity = (
                    candidate
                    -
                    state.last_output_angle
                ) / delta_input


                velocity_change = abs(
                    velocity
                    -
                    state.output_velocity
                )

            else:

                velocity_change = 0.0


            if (
                (
                    candidate
                    -
                    state.last_output_angle
                )
                *
                state.direction
                >= 0
            ):

                direction_penalty = 0.0

            else:

                direction_penalty = 1.0


            return (
                prediction_error
                +
                5.0 * output_change
                +
                20.0 * velocity_change
                +
                direction_penalty
            )


        return min(
            brackets,
            key=score,
        )