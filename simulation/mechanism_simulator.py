"""
simulation/mechanism_simulator.py

Simulation of complete mechanisms.
"""

from __future__ import annotations

from mechanics.mechanism import Mechanism
from simulation.motion_range import MotionRange
from simulation.simulation_result import SimulationResult
from simulation.stage_simulator import StageSimulator
from solver.solver_precision import SolverPrecision


class MechanismSimulator:
    """
    Simulates every stage of a mechanism.

    The simulator is configured with a motion range and can
    repeatedly evaluate different mechanisms using identical
    simulation settings.

    Optionally, simulation can be limited to the first N stages.
    """

    def __init__(
        self,
        *,
        motion: MotionRange,
        stage_simulator: StageSimulator | None = None,
        stage_limit: int | None = None,
        precision: SolverPrecision | None = None,
    ) -> None:

        if (
            stage_limit is not None
            and stage_limit < 1
        ):
            raise ValueError(
                "stage_limit must be greater than zero"
            )

        self._motion = motion

        self._precision = precision

        # Existing stage simulator owns its solver configuration.
        self._stage_simulator = (
            stage_simulator
            if stage_simulator is not None
            else StageSimulator(
                precision=precision,
            )
        )
        
        self._stage_limit = stage_limit


    @property
    def motion(self) -> MotionRange:
        """
        Motion range used for all simulations.
        """

        return self._motion

    @property
    def precision(self) -> SolverPrecision | None:
        """
        Solver precision configuration.

        None means that the default precision
        configuration is used by the stage solver.
        """

        return self._precision

    @property
    def stage_limit(self) -> int | None:
        """
        Maximum number of simulated stages.

        None means all stages are simulated.
        """

        return self._stage_limit


    def simulate(
        self,
        mechanism: Mechanism,
    ) -> tuple[SimulationResult, ...]:
        """
        Simulate stages of a mechanism.

        Parameters
        ----------
        mechanism:
            Mechanism to simulate.

        Returns
        -------
        tuple[SimulationResult, ...]
            One SimulationResult per simulated stage.
        """

        stages = mechanism.stages

        if self._stage_limit is not None:

            stages = stages[:self._stage_limit]


        return tuple(
            self._stage_simulator.run(
                stage=stage,
                motion=self._motion,
            )
            for stage in stages
        )
        