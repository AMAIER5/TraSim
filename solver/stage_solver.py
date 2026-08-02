"""
solver/stage_solver.py

High level solver for one mechanical stage.
"""

from __future__ import annotations

from mechanics.stage import Stage
from solver.angle_solver import AngleSolver
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
        angle_solver: AngleSolver | None = None,
    ):
        self.stage = stage

        self.angle_solver = (
            angle_solver
            if angle_solver is not None
            else AngleSolver()
        )

    def solve(
        self,
        *,
        input_angle: float,
        state: SolverState,
    ) -> tuple[SolverResult, SolverState]:
        """
        Solve one simulation step.

        Returns
        -------
        result:
            Solver result.

        state:
            Updated solver state.
        """

        result = self.angle_solver.solve(
            stage=self.stage,
            input_angle=input_angle,
            state=state,
        )

        if not result.success:
            return result, state

        new_state = state.next(
            input_angle=input_angle,
            output_angle=result.angle,
        )

        return result, new_state

    def get_stats(self) -> dict[str, int]:
        """
        Return performance statistics collected by
        the underlying AngleSolver.
        """

        return self.angle_solver.get_stats()

    def reset_stats(self) -> None:
        """
        Reset performance statistics collected by
        the underlying AngleSolver.
        """

        self.angle_solver.reset_stats()