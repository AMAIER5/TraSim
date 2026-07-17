"""
simulation/motion_range.py

Defines the input angle sequence for a simulation run.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.tolerance import ANGLE_TOLERANCE


@dataclass(frozen=True, slots=True)
class MotionRange:
    """
    Defines a one-dimensional angular simulation range.

    Angles are stored internally in radians.

    Parameters
    ----------
    start_angle:
        Initial angle [rad].

    max_angle:
        Maximum absolute travel angle [rad].

    step:
        Angular increment [rad].

    direction:
        +1 increasing angle.
        -1 decreasing angle.
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

    def __iter__(self):

        current = self.start_angle

        travelled = 0.0

        while travelled <= self.max_angle + ANGLE_TOLERANCE:

            yield current

            current += (
                self.direction
                * self.step
            )

            travelled += self.step