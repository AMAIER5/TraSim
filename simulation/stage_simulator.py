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
from solver.solver_precision import SolverPrecision
from solver.solver_result import SolverResult
from solver.stage_solver import StageSolver


class SolverProtocol(Protocol):
    """
    Runtime interface required from a stage solver.
    """

    def __init__(
        self,
        stage: Stage,
        *,
        precision: SolverPrecision | None = None,
    ) -> None:
        ...

    def solve(
        self,
        *,
        input_angle: float,
    ) -> SolverResult:
        ...

    def get_stats(self) -> dict[str, int]:
        ...

    def reset_stats(self) -> None:
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
        precision: SolverPrecision | None = None,
    ) -> None:

        self._solver_type = solver_type
        self._precision = precision
        self._solvers: list[SolverProtocol] = []

    @property
    def solvers(self) -> tuple[SolverProtocol, ...]:
        """
        Return all solver instances created so far.

        Used for diagnostics and performance statistics.
        """

        return tuple(self._solvers)

    def run(
        self,
        *,
        stage: Stage,
        motion: MotionProvider,
    ) -> SimulationResult:
        """
        Simulate a stage over the specified motion provider.
        """

        solver = self._solver_type(
            stage,
            precision=self._precision,
        )

        self._solvers.append(solver)

        input_angles: list[float] = []
        output_angles: list[float] = []

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

            result = solver.solve(
                input_angle=input_angle,
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

            if previous_output is not None:
                motion.feedback(
                    output_delta=result.angle - previous_output,
                )

            previous_output = result.angle

        return SimulationResult(
            input_angles=tuple(input_angles),
            output_angles=tuple(output_angles),
            success=True,
        )