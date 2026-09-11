"""
simulation/point_motion.py

Motion provider that yields an explicit, fixed sequence of
input angles.

Unlike MotionRange (which generates an equidistant grid from a
start angle, step size and travel range), PointMotion reproduces an
arbitrary, possibly non-equidistant list of input angles exactly.
This is used to simulate a mechanism precisely at the support
points of a desired target curve (e.g. the 11 sampled points of a
CSV target curve), instead of an evenly spaced grid.

The provider is immutable and can be iterated multiple times;
every iteration yields the same angles in the same order, so the
output of a stage can feed the next stage in a kinematic chain
(see simulation.result_motion_provider.ResultMotionProvider).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PointMotion:
    """
    Fixed input-angle sequence for a simulation run.

    Angles are stored internally in radians and must be a
    non-empty, monotonically non-decreasing or non-increasing
    sequence so the stage solver can follow the motion.

    The compatible ``start_angle``, ``max_angle``, ``step`` and
    ``direction`` properties allow PointMotion to be used as a
    drop-in replacement for MotionRange where a motion identity is
    required (e.g. MechanismOptimizer cache keys).
    """

    angles: tuple[float, ...]

    def __post_init__(self) -> None:

        if not self.angles:
            raise ValueError(
                "angles must contain at least one value"
            )

    def __iter__(self) -> Iterator[float]:
        """
        Generate the stored input angles.

        The sequence is fixed; repeated iteration yields the
        same angles in the same order.
        """

        yield from self.angles

    def feedback(
        self,
        *,
        output_delta: float,
    ) -> None:
        """
        Receive simulation feedback.

        A fixed point sequence does not adapt its angles.
        """

        pass

    @property
    def start_angle(self) -> float:
        """
        First angle of the sequence.

        Compatible with MotionRange.start_angle so motion
        providers can be used interchangeably for cache keys.
        """

        return self.angles[0]

    @property
    def max_angle(self) -> float:
        """
        Travel range of the sequence (last - first).

        Compatible with MotionRange.max_angle, which stores the
        travelled distance rather than the absolute end angle.
        """

        return self.angles[-1] - self.angles[0]

    @property
    def step(self) -> float:
        """
        Representative step size of the sequence.

        For a non-equidistant sequence this is a sentinel value
        (0.0) that distinguishes PointMotion from equidistant
        MotionRange instances in cache keys while remaining stable
        for a given angle list.
        """

        return 0.0

    @property
    def direction(self) -> int:
        """
        Direction of the sequence (+1 ascending, -1 descending).

        Compatible with MotionRange.direction.
        """

        if self.angles[-1] >= self.angles[0]:
            return 1

        return -1
