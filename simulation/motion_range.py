"""
simulation/motion_range.py

Defines the input angle sequence for a simulation run.

Issue #8 (and #12): The original ``__iter__`` accumulated
``travelled += self.step`` in a while loop, comparing the
float sum against ``max_angle + ANGLE_TOLERANCE``.  While
this works for moderate iteration counts, it is
FP-sensitive for very large counts and conceptually
confusing: ``travelled`` tracks the *distance* moved,
not the *actual angle*.

The fix replaces the accumulation with integer-counted
iteration: the number of steps is computed once via
``count`` (integer arithmetic), and each yielded angle
is derived from ``start_angle + direction * i * step``
instead of accumulating ``current``.

This also documents that ``max_angle == 0.0`` yields
exactly one point (``start_angle``) — intentional, used
by tests that need a single-point simulation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from core.tolerance import ANGLE_TOLERANCE


@dataclass(frozen=True, slots=True)
class MotionRange:
    """
    Defines a one-dimensional angular simulation range.

    Angles are stored internally in radians.

    The sequence contains::

        start_angle,
        start_angle + step,
        start_angle + 2*step,
        ...,
        start_angle + N*step

    where N is the largest integer such that
    ``N * step <= max_angle + ANGLE_TOLERANCE``.

    When ``max_angle`` is 0.0 exactly one angle
    (``start_angle``) is yielded.  This is intentional and
    used by tests that need a single-point simulation.
    """

    start_angle: float

    max_angle: float

    step: float

    direction: int = 1

    def __post_init__(self) -> None:

        if self.direction not in (-1, 1):
            raise ValueError(
                "direction must be +1 or -1"
            )

        if self.step <= 0:
            raise ValueError(
                "step must be positive"
            )

        if self.max_angle < 0:
            raise ValueError(
                "max_angle must not be negative"
            )

    @property
    def count(self) -> int:
        """
        Number of angles yielded by this motion range.

        Computed via integer arithmetic to avoid
        floating-point accumulation drift.
        """

        if self.max_angle == 0.0:
            return 1

        return (
            int(
                math.floor(
                    self.max_angle
                    / self.step
                    + ANGLE_TOLERANCE
                )
            )
            + 1
        )

    def __iter__(self) -> Iterator[float]:
        """
        Generate input angles.

        Issue #8: The angle at position *i* is::

            start_angle + direction * i * step

        Using ``i * step`` instead of accumulating
        ``current += step`` avoids floating-point drift
        for large iteration counts and eliminates the
        confusing ``travelled`` accumulator entirely.
        """

        for i in range(self.count):

            yield (
                self.start_angle
                + self.direction * i * self.step
            )

    def feedback(
        self,
        *,
        output_delta: float,
    ) -> None:
        """
        Receive simulation feedback.

        Fixed motion ranges do not adapt their step size.
        """
        pass