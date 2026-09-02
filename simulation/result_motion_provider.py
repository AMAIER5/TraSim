"""
simulation/result_motion_provider.py

Motion provider based on the output of a previous simulation stage.
"""

from __future__ import annotations

from collections.abc import Iterator

from simulation.simulation_result import SimulationResult


class ResultMotionProvider:
    """
    Provides the output motion of a previous simulation stage
    as input motion for the next stage.

    The provider is immutable and can be iterated multiple times.
    """

    def __init__(
        self,
        result: SimulationResult,
    ) -> None:

        self._angles = result.output_angles


    def __iter__(self) -> Iterator[float]:
        """
        Generate input angles from previous stage output.
        """

        yield from self._angles


    def feedback(
        self,
        *,
        output_delta: float,
    ) -> None:
        """
        Receive simulation feedback.

        Result based motion is not adaptive.
        """

        pass