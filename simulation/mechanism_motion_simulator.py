"""
simulation/mechanism_motion_simulator.py

Runs a complete motion simulation
of a multi-stage mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from mechanics.mechanism import Mechanism

from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange


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
    ) -> None:

        self._mechanism = mechanism

    def run(
        self,
        motion: MotionRange,
    ) -> MechanismMotionResult:
        """
        Run complete motion.
        """

        simulator = MechanismSimulator(
            motion=motion,
        )

        stage_results = simulator.simulate(
            self._mechanism,
        )

        if not stage_results:

            return MechanismMotionResult(
                input_angles=(),
                stage_outputs=(),
                success=True,
            )

        input_angles = (
            stage_results[0].input_angles
        )

        success = all(
            result.success
            for result in stage_results
        )

        blocked_at = next(
            (
                result.blocked_at
                for result in stage_results
                if not result.success
            ),
            None,
        )

        stage_outputs = tuple(
            tuple(
                result.output_angles[index]
                for result in stage_results
            )
            for index in range(
                len(input_angles)
            )
        )

        return MechanismMotionResult(
            input_angles=input_angles,
            stage_outputs=stage_outputs,
            success=success,
            blocked_at=blocked_at,
        )