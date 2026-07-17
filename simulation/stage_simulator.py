"""
simulation/stage_simulator.py

Runs a complete motion simulation for one stage.
"""

from __future__ import annotations

from mechanics.stage import Stage

from simulation.motion_range import MotionRange
from simulation.simulation_result import SimulationResult

from solver.solver_state import SolverState
from solver.stage_solver import StageSolver


class StageSimulator:
    """
    Simulates one mechanical stage over a motion range.
    """

    def __init__(
        self,
        stage: Stage,
        solver: StageSolver | None = None,
    ):
        self.stage = stage

        self.solver = (
            solver
            if solver is not None
            else StageSolver(stage)
        )

    def run(
        self,
        motion: MotionRange,
    ) -> SimulationResult:
        """
        Execute simulation.

        Stops when solver cannot find
        a valid next position.
        """

        input_angles: list[float] = []

        output_angles: list[float] = []

        state = SolverState(
            last_input_angle=motion.start_angle,
            last_output_angle=motion.start_angle,
        )

        for input_angle in motion:

            result, state = self.solver.solve(
                input_angle=input_angle,
                state=state,
            )

            if not result.success:

                return SimulationResult(
                    input_angles=tuple(input_angles),
                    output_angles=tuple(output_angles),
                    success=False,
                    blocked_at=input_angle,
                )

            input_angles.append(
                input_angle
            )

            output_angles.append(
                result.angle
            )

        return SimulationResult(
            input_angles=tuple(input_angles),
            output_angles=tuple(output_angles),
            success=True,
        )