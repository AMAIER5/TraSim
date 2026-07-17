"""
simulation/mechanism_motion_simulator.py

Runs a complete motion simulation
of a multi-stage mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from mechanics.mechanism import Mechanism

from simulation.motion_range import MotionRange
from simulation.mechanism_simulator import (
    MechanismSimulator,
)

from solver.mechanism_state import MechanismState


@dataclass(frozen=True, slots=True)
class MechanismMotionResult:
    """
    Complete motion result.

    stage_outputs contains one tuple per input angle.
    Each tuple contains the output angle of every stage.
    """

    input_angles: tuple[float, ...]

    stage_outputs: tuple[
        tuple[float, ...],
        ...
    ]

    success: bool

    blocked_at: float | None = None


class MechanismMotionSimulator:
    """
    Executes a complete motion of a mechanism.
    """

    def __init__(
        self,
        mechanism: Mechanism,
    ):

        self.mechanism = mechanism

        self.simulator = MechanismSimulator(
            mechanism
        )

    def run(
        self,
        motion: MotionRange,
    ) -> MechanismMotionResult:
        """
        Run complete motion.
        """

        input_angles = []

        stage_outputs = []

        state = MechanismState(
            stage_states=()
        )

        for angle in motion:

            result = self.simulator.solve(
                input_angle=angle,
            )

            if not result.success:

                return MechanismMotionResult(
                    input_angles=tuple(
                        input_angles
                    ),
                    stage_outputs=tuple(
                        stage_outputs
                    ),
                    success=False,
                    blocked_at=angle,
                )

            input_angles.append(
                angle
            )

            stage_outputs.append(
                result.output_angles
            )

        return MechanismMotionResult(
            input_angles=tuple(
                input_angles
            ),
            stage_outputs=tuple(
                stage_outputs
            ),
            success=True,
        )