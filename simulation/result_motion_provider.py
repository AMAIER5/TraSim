"""
simulation/result_motion_provider.py

Motion provider based on the output of a previous
simulation stage.

Issue (coupled lever angle mapping): the provider now
supports an ``angle_offset``.  When the input lever of
the following stage is a lever that is coupled to the
previous stage's output lever, its lever angle differs
from the previous lever angle by a constant (the
coupling offset).  The offset is applied to every
provided angle so the following stage receives angles
in ITS OWN lever-angle frame, as required by the
kinematic model.
"""

from __future__ import annotations

from collections.abc import Iterator

from simulation.simulation_result import SimulationResult


class ResultMotionProvider:
    """
    Provides the output motion of a previous simulation stage
    as input motion for the next stage.

    The provider is immutable and can be iterated multiple
    times.

    Parameters
    ----------
    result:
        Simulation result of the previous stage.

    angle_offset:
        Constant offset added to every previous-stage output
        angle [rad].  Zero for a plain chain where the next
        stage's input lever IS the previous stage's output
        lever; the coupling offset
        (``angle_start - coupled.angle_start``) when the next
        stage's input lever is coupled to the previous output
        lever.
    """

    def __init__(
        self,
        result: SimulationResult,
        angle_offset: float = 0.0,
    ) -> None:

        self._angles = result.output_angles
        self._angle_offset = angle_offset

    def __iter__(self) -> Iterator[float]:
        """
        Generate input angles from previous stage output.
        """

        for angle in self._angles:
            yield angle + self._angle_offset

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
