"""
simulation/stage_simulator.py

Simulates one mechanical stage over a motion provider.

Motion generation is delegated to MotionProvider.
The simulator itself is independent of fixed or adaptive sampling.
"""

from __future__ import annotations

from typing import Protocol

from mechanics.stage import Stage
from simulation.motion_provider import MotionProvider
from simulation.simulation_result import SimulationResult
from solver.objective import stage_error
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
    Motion generation is delegated to MotionProvider.
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
        motion: MotionProvider,
    ) -> SimulationResult:
        """
        Simulate a stage over the specified motion provider.

        The solver follows the physical motion branch using SolverState.
        """

        solver = self._solver_type(stage)

        input_angles: list[float] = []
        output_angles: list[float] = []

        state = SolverState.initial(
            input_angle=stage.input_angle,
            output_angle=stage.output_angle,
        )

        previous_output: float | None = None

        reference_error = abs(
            stage_error(
                stage,
                stage.input_angle,
                stage.output_angle,
            )
        )

        if reference_error > 1e-9:
            raise ValueError(
                f"Invalid stage reference geometry: {reference_error}"
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

            input_angles.append(
                input_angle
            )

            output_angles.append(
                result.angle
            )

            if previous_output is not None:

                output_delta = (
                    result.angle
                    -
                    previous_output
                )

                motion.feedback(
                    output_delta=output_delta
                )

            previous_output = result.angle

        return SimulationResult(
            input_angles=tuple(input_angles),
            output_angles=tuple(output_angles),
            success=True,
        )