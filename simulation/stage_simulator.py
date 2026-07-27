"""
simulation/stage_simulator.py

Simulates one mechanical stage over a motion range.
"""

from __future__ import annotations

from typing import Protocol

from mechanics.stage import Stage

from simulation.motion_range import MotionRange
from simulation.simulation_result import SimulationResult

from solver.solver_result import SolverResult
from solver.solver_state import SolverState
from solver.stage_solver import StageSolver


class SolverProtocol(Protocol):
    """
    Protocol implemented by stage solvers.
    """

    def __init__(
        self,
        stage: Stage,
    ) -> None:
        ...

    def solve(
        self,
        *,
        input_angle: float,
        state: SolverState,
    ) -> tuple[SolverResult, SolverState]:
        ...


class StageSimulator:
    """
    Stateless simulator for one mechanical stage.

    A new solver instance is created for every simulation run.
    """

    def __init__(
        self,
        *,
        solver_type: type[SolverProtocol] = StageSolver,
    ) -> None:
        self._solver_type = solver_type

    def run(
        self,
        *,
        stage: Stage,
        motion: MotionRange,
    ) -> SimulationResult:
        """
        Simulate a stage over the specified motion range.

        Parameters
        ----------
        stage:
            Mechanical stage to simulate.

        motion:
            Sequence of input angles.

        Returns
        -------
        SimulationResult
            Simulation result containing input/output angles and
            solver status.
        """

        solver = self._solver_type(stage)

        input_angles: list[float] = []
        output_angles: list[float] = []

        state = SolverState(
            last_input_angle=motion.start_angle,
            last_output_angle=motion.start_angle,
        )

        for input_angle in motion:

            result, state = solver.solve(
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

            input_angles.append(input_angle)
            output_angles.append(result.angle)

        return SimulationResult(
            input_angles=tuple(input_angles),
            output_angles=tuple(output_angles),
            success=True,
        )