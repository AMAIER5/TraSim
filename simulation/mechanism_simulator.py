"""
simulation/mechanism_simulator.py

Simulation of complete mechanisms.
"""

from __future__ import annotations

from mechanics.mechanism import Mechanism
from simulation.motion_range import MotionRange
from simulation.simulation_result import SimulationResult
from simulation.stage_simulator import StageSimulator


class MechanismSimulator:
    """
    Simulates every stage of a mechanism.

    The simulator is configured with a motion range and can
    repeatedly evaluate different mechanisms using identical
    simulation settings.
    """

    def __init__(
        self,
        *,
        motion: MotionRange,
        stage_simulator: StageSimulator | None = None,
    ) -> None:
        self._motion = motion
        self._stage_simulator = (
            stage_simulator
            if stage_simulator is not None
            else StageSimulator()
        )

    @property
    def motion(self) -> MotionRange:
        """
        Motion range used for all simulations.
        """
        return self._motion

    def simulate(
        self,
        mechanism: Mechanism,
    ) -> tuple[SimulationResult, ...]:
        """
        Simulate every stage of a mechanism.

        Parameters
        ----------
        mechanism:
            Mechanism to simulate.

        Returns
        -------
        tuple[SimulationResult, ...]
            One SimulationResult per stage.
        """

        return tuple(
            self._stage_simulator.run(
                stage=stage,
                motion=self._motion,
            )
            for stage in mechanism.stages
        )