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
- cached input objective evaluation

Branch selection considers:
- prediction distance
- output angle jump
- velocity change
- motion direction
"""

from __future__ import annotations

import math

from mechanics.stage import Stage
from solver.objective import create_stage_objective
from solver.root_solver import RootSolver
from solver.solver_result import SolverResult
from solver.solver_state import SolverState
from solver.solver_precision import SolverPrecision


class AngleSolver:
    """
    Solve output angle for a given input angle.

    Multiple mathematical solutions may exist.

    The solver keeps the physical motion branch by using
    SolverState prediction and local kinematic continuity.
    """

    def __init__(
        self,
        stage: Stage,
        *,
        precision: SolverPrecision | None = None,
        search_min: float = math.radians(-180),
        search_max: float = math.radians(180),
        bracket_step: float | None = None,
        search_window: float | None = None,
        tolerance: float | None = None,
        max_iterations: int | None = None,
        reuse_factor: float = 3.0,
        reuse_min_window: float = math.radians(2),
        reuse_max_window: float = math.radians(15),
    ) -> None:

        self.stage = stage

        if precision is None:
            precision = SolverPrecision()

        if (
            bracket_step is not None
            or search_window is not None
            or tolerance is not None
            or max_iterations is not None
        ):

            precision = SolverPrecision(
                tolerance=(
                    tolerance
                    if tolerance is not None
                    else precision.tolerance
                ),
                max_iterations=(
                    max_iterations
                    if max_iterations is not None
                    else precision.max_iterations
                ),
                bracket_step=(
                    bracket_step
                    if bracket_step is not None
                    else precision.bracket_step
                ),
                search_window=(
                    search_window
                    if search_window is not None
                    else precision.search_window
                ),
            )

        self.precision = precision
        self.search_min = search_min
        self.search_max = search_max

        self.reuse_factor = reuse_factor
        self.reuse_min_window = reuse_min_window
        self.reuse_max_window = reuse_max_window

        self._last_root: float | None = None
        self._last_bracket_width: float = self.precision.bracket_step

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
            "brent_iterations": 0,
        }

    def solve(
        self,
        input_angle: float,
        state: SolverState,
    ) -> SolverResult:

        residual = create_stage_objective(
            self.stage,
            input_angle,
        )

        predicted = state.predict_output(
            input_angle
        )

        brackets = []

        # --------------------------------------------------------------
        # First try: reuse previous solution branch
        # --------------------------------------------------------------

        if self._last_root is not None:

            self.stats[
                "adaptive_attempts"
            ] += 1

            window = min(
                max(
                    self._last_bracket_width
                    * self.reuse_factor,
                    self.reuse_min_window,
                ),
                self.reuse_max_window,
            )

            brackets = RootSolver.find_all_brackets_around(
                function=residual,
                center=predicted,
                window=window,
                step=self.precision.bracket_step,
            )

            # Fix #4: filter fast-path brackets to allowed output range
            brackets = self._filter_to_allowed_range(brackets)

            if brackets:
                self.stats[
                    "adaptive_success"
                ] += 1
            else:
                self.stats[
                    "adaptive_failure"
                ] += 1

        # --------------------------------------------------------------
        # Local search around predicted branch
        # --------------------------------------------------------------

        if not brackets:

            self.stats[
                "local_searches"
            ] += 1

            brackets = RootSolver.find_all_brackets_around(
                function=residual,
                center=predicted,
                window=self.precision.search_window,
                step=self.precision.bracket_step,
            )

            # Fix #4: filter fast-path brackets to allowed output range
            brackets = self._filter_to_allowed_range(brackets)

        # --------------------------------------------------------------
        # Full-range search (fallback only)
        # Fix #3: guard with 'if not brackets' so the full-range
        # search does not overwrite results from the adaptive
        # and local searches above.
        # --------------------------------------------------------------

        allowed_min = max(
            self.search_min,
            self.stage.output_angle_min,
        )

        allowed_max = min(
            self.search_max,
            self.stage.output_angle_max,
        )

        if not brackets:

            self.stats[
                "fallback_searches"
            ] += 1

            if allowed_min <= allowed_max:

                brackets = RootSolver.find_all_brackets(
                    function=residual,
                    minimum=allowed_min,
                    maximum=allowed_max,
                    step=self.precision.bracket_step,
                )

            else:

                brackets = []

        self.stats[
            "brackets_found"
        ] += len(brackets)

        # --------------------------------------------------------------
        # No valid bracket inside output limits
        # Diagnose whether root exists outside limits
        # --------------------------------------------------------------

        if not brackets:

            diagnostic_brackets = RootSolver.find_all_brackets(
                function=residual,
                minimum=self.search_min,
                maximum=self.search_max,
                step=self.precision.bracket_step,
            )

            if diagnostic_brackets:

                self.stats["blocked"] += 1

                return SolverResult(
                    success=False,
                    angle=float("nan"),
                    residual=float("inf"),
                    iterations=0,
                    reason="output_angle_limit",
                )

            self.stats["blocked"] += 1

            return SolverResult(
                success=False,
                angle=float("nan"),
                residual=float("inf"),
                iterations=0,
                reason="blocked",
            )

        # --------------------------------------------------------------
        # Select branch
        # --------------------------------------------------------------

        if len(brackets) == 1:

            bracket = brackets[0]

            self.stats[
                "single_bracket_fast_path"
            ] += 1

        else:

            bracket = self._select_branch(
                brackets,
                reference_angle=predicted,
                state=state,
                input_angle=input_angle,
            )

            self.stats[
                "multi_bracket_selection"
            ] += 1

        left, right, bracket_iterations = bracket

        # --------------------------------------------------------------
        # Brent solve
        # --------------------------------------------------------------

        try:

            angle, value, solver_iterations = (
                RootSolver.solve_brent(
                    function=residual,
                    left=left,
                    right=right,
                    tolerance=self.precision.tolerance,
                    max_iterations=self.precision.max_iterations,
                )
            )

            self.stats[
                "brent_iterations"
            ] += solver_iterations

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
            <= self.precision.tolerance
        )

        if success:

            self.stats["solved"] += 1

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
                + solver_iterations
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
        self._last_bracket_width = self.precision.bracket_step

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
                - reference_angle
            )

            if state.direction == 0:
                return prediction_error

            output_change = abs(
                candidate
                - state.last_output_angle
            )

            delta_input = (
                input_angle
                - state.last_input_angle
            )

            if abs(delta_input) > 1e-12:

                velocity = (
                    candidate
                    - state.last_output_angle
                ) / delta_input

                velocity_change = abs(
                    velocity
                    - state.output_velocity
                )

            else:

                velocity_change = 0.0

            direction_penalty = 0.0

            if (
                (
                    candidate
                    - state.last_output_angle
                )
                * state.direction
                < 0
            ):
                direction_penalty = 1.0

            return (
                prediction_error
                + 5.0 * output_change
                + 20.0 * velocity_change
                + direction_penalty
            )

        return min(
            brackets,
            key=score,
        )

    def _filter_to_allowed_range(
        self,
        brackets: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, int]]:
        """
        Remove brackets whose center lies outside the
        stage's allowed output angle range.

        Fix #4: The adaptive and local searches around the
        predicted position are not clipped to the stage's
        [output_angle_min, output_angle_max] range by
        RootSolver.find_all_brackets_around.  A bracket found
        there could yield a Brent solution outside the
        mechanical limits.  This filter discards such brackets
        so the solver falls through to the full-range search.
        """

        allowed_min = max(
            self.search_min,
            self.stage.output_angle_min,
        )

        allowed_max = min(
            self.search_max,
            self.stage.output_angle_max,
        )

        return [
            (left, right, n)
            for left, right, n in brackets
            if allowed_min <= (left + right) / 2.0 <= allowed_max
        ]