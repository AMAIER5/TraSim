"""
solver/stage_solver.py

High level solver for one mechanical stage.
"""

from __future__ import annotations

from mechanics.stage import Stage
from solver.angle_solver import AngleSolver
from solver.solver_precision import SolverPrecision
from solver.solver_result import SolverResult
from solver.solver_state import SolverState


class StageSolver:
    """
    Solver wrapper for one kinematic stage.

    Handles:
    - stage reference
    - solver state
    - angle solving
    """

    def __init__(
        self,
        stage: Stage,
        *,
        precision: SolverPrecision | None = None,
        angle_solver: AngleSolver | None = None,
    ) -> None:

        self.stage = stage

        self.angle_solver = (
            angle_solver
            if angle_solver is not None
            else AngleSolver(
                stage,
                precision=precision,
            )
        )

        self._state: SolverState | None = None

    def solve(
        self,
        *,
        input_angle: float,
    ) -> SolverResult:
        """
        Solve one simulation step.
        """

        if self._state is None:
            self._state = SolverState.initial(
                input_angle=self.stage.input_angle,
                output_angle=self.stage.output_angle,
            )

        result = self.angle_solver.solve(
            input_angle=input_angle,
            state=self._state,
        )

        if result.success:
            self._state = self._state.next(
                input_angle=input_angle,
                output_angle=result.angle,
            )

        return result

    def get_stats(self) -> dict[str, int]:
        """
        Return performance statistics collected by the
        underlying AngleSolver.
        """

        return self.angle_solver.get_stats()

    def reset_stats(self) -> None:
        """
        Reset performance statistics collected by the
        underlying AngleSolver.
        """

        self.angle_solver.reset_stats()