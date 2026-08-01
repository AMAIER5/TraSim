"""
simulation/adaptive_motion_range.py

Adaptive input angle generation for simulation runs.

Implements adaptive motion feedback through feedback().
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class AdaptiveMotionRange:
    """
    Defines an adaptive one-dimensional angular simulation range.

    Angles are stored internally in radians.

    The step size is adjusted based on output motion feedback.
    """

    start_angle: float

    end_angle: float

    initial_step: float = math.radians(5)

    min_step: float = math.radians(0.25)

    max_step: float = math.radians(10)

    max_output_delta: float = math.radians(5)

    current_step: float = field(
        init=False
    )

    def __post_init__(self) -> None:

        if self.initial_step <= 0:
            raise ValueError(
                "initial_step must be positive"
            )

        if self.min_step <= 0:
            raise ValueError(
                "min_step must be positive"
            )

        if self.max_step < self.min_step:
            raise ValueError(
                "max_step must be >= min_step"
            )

        self.current_step = self.initial_step

    def __iter__(self) -> Iterator[float]:
        """
        Generate input angles.

        The current step size is controlled internally.
        """

        current_angle = self.start_angle

        direction = (
            1
            if self.end_angle >= self.start_angle
            else -1
        )

        while True:

            if direction > 0:

                if current_angle > self.end_angle:
                    break

            else:

                if current_angle < self.end_angle:
                    break

            yield current_angle

            current_angle += (
                direction
                *
                self.current_step
            )

    def feedback(
        self,
        *,
        output_delta: float,
    ) -> None:
        """
        Update adaptive step size from simulation feedback.

        Parameters
        ----------
        output_delta:
            Difference between current and previous output angle [rad].
        """

        if abs(output_delta) > self.max_output_delta:

            self.current_step *= 0.5

        elif abs(output_delta) < self.max_output_delta * 0.25:

            self.current_step *= 1.5

        self.current_step = max(
            self.min_step,
            min(
                self.current_step,
                self.max_step,
            )
        )