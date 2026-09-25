"""
simulation/mechanism_simulator.py

Simulation of complete mechanisms.
"""

from __future__ import annotations

from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from simulation.motion_provider import MotionProvider
from simulation.motion_range import MotionRange
from simulation.result_motion_provider import ResultMotionProvider
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

        results = []

        motion: MotionProvider = self._motion

        for index, stage in enumerate(stages):

            result = self._stage_simulator.run(
                stage=stage,
                motion=motion,
            )

            results.append(result)

            motion = ResultMotionProvider(
                result,
                angle_offset=(
                    self._coupling_offset(
                        stages,
                        index,
                    )
                ),
            )

        return tuple(results)

    @staticmethod
    def _coupling_offset(
        stages: tuple[Stage, ...],
        index: int,
    ) -> float:
        """
        Lever-angle offset between stage ``index`` and the
        following stage.

        The input lever of the following stage is either the
        same physical lever as the current stage's output lever
        (plain chain, offset 0) or a lever coupled to it.  For a
        coupled lever the lever angle differs from the coupled
        lever's angle by a constant: the difference of the two
        reference lever angles.  The offset shifts the current
        stage's output angles into the following stage's input
        lever-angle frame, so angle limits and kinematics of
        the following stage are evaluated in the correct frame.
        Without the offset, a coupled intermediate lever would
        be simulated (and drawn) at the wrong angle and could
        leave its admissible lever-angle segment unnoticed.
        """

        if index + 1 >= len(stages):
            return 0.0

        current = stages[index]
        following = stages[index + 1]

        if (
            following.input_lever
            is current.output_lever
        ):
            return 0.0

        return (
            following.input_angle
            - current.output_angle
        )