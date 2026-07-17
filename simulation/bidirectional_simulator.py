"""
simulation/bidirectional_simulator.py

Runs a stage simulation in both angular directions.
"""

from __future__ import annotations

from mechanics.stage import Stage

from simulation.motion_range import MotionRange
from simulation.simulation_result import SimulationResult
from simulation.stage_simulator import StageSimulator


class BidirectionalSimulator:
    """
    Simulates a stage starting from a given angle
    in negative and positive direction.
    """

    def __init__(
        self,
        stage: Stage,
        simulator: StageSimulator | None = None,
    ):

        self.stage = stage

        self.simulator = (
            simulator
            if simulator is not None
            else StageSimulator(stage)
        )

    def run(
        self,
        *,
        start_angle: float,
        max_angle: float,
        step: float,
    ) -> SimulationResult:
        """
        Run simulation in both directions.

        The first result entry is always the start position.

        Angles are absolute angles in radians.
        """

        negative_motion = MotionRange(
            start_angle=start_angle,
            max_angle=max_angle,
            step=step,
            direction=-1,
        )

        positive_motion = MotionRange(
            start_angle=start_angle,
            max_angle=max_angle,
            step=step,
            direction=1,
        )

        negative_result = self.simulator.run(
            negative_motion
        )

        positive_result = self.simulator.run(
            positive_motion
        )

        input_angles = (
            negative_result.input_angles
            +
            positive_result.input_angles[1:]
        )

        output_angles = (
            negative_result.output_angles
            +
            positive_result.output_angles[1:]
        )

        success = (
            negative_result.success
            and positive_result.success
        )

        blocked_at = None

        if not negative_result.success:

            blocked_at = negative_result.blocked_at

        elif not positive_result.success:

            blocked_at = positive_result.blocked_at

        return SimulationResult(
            input_angles=input_angles,
            output_angles=output_angles,
            success=success,
            blocked_at=blocked_at,
        )