"""
simulation/mechanism_simulator.py

Simulator for complete multi-stage mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

from mechanics.mechanism import Mechanism

from solver.solver_state import SolverState
from solver.stage_solver import StageSolver


@dataclass(frozen=True, slots=True)
class MechanismSimulationResult:
    """
    Result of a complete mechanism calculation.
    """

    stage_inputs: tuple[float, ...]

    output_angles: tuple[float, ...]

    success: bool


class MechanismSimulator:
    """
    Simulates a chain of connected stages.
    """

    def __init__(
        self,
        mechanism: Mechanism,
    ):

        self.mechanism = mechanism

        self.stage_solvers = tuple(
            StageSolver(stage)
            for stage in mechanism.stages
        )

    def solve(
        self,
        *,
        input_angle: float,
    ) -> MechanismSimulationResult:
        """
        Solve all stages sequentially.
        """

        stage_inputs = []

        output_angles = []

        current_angle = input_angle

        for solver in self.stage_solvers:

            stage_inputs.append(
                current_angle
            )

            state = SolverState(
                last_input_angle=current_angle,
                last_output_angle=current_angle,
            )

            result, _ = solver.solve(
                input_angle=current_angle,
                state=state,
            )

            if not result.success:

                return MechanismSimulationResult(
                    stage_inputs=tuple(stage_inputs),
                    output_angles=tuple(output_angles),
                    success=False,
                )

            output_angles.append(
                result.angle
            )

            current_angle = result.angle

        return MechanismSimulationResult(
            stage_inputs=tuple(stage_inputs),
            output_angles=tuple(output_angles),
            success=True,
        )