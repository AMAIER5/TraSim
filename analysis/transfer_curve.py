"""
analysis/transfer_curve.py

Representation of an input/output angle relationship.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransferCurve:
    """
    Represents a kinematic transfer function.

    Input and output angles must be ordered
    and have equal length.

    The input angles may be either ascending
    or descending.
    """

    input_angles: tuple[float, ...]

    output_angles: tuple[float, ...]

    def __post_init__(self) -> None:

        if len(
            self.input_angles
        ) != len(
            self.output_angles
        ):

            raise ValueError(
                "input and output length mismatch"
            )

        if len(
            self.input_angles
        ) < 2:

            raise ValueError(
                "at least two points required"
            )

    def output_at(
        self,
        input_angle: float,
    ) -> float:
        """
        Linear interpolation of output angle.

        The input range must be covered by
        the curve.

        Supports both ascending and descending
        input angle sequences.
        """

        eps = 1e-12

        minimum = min(
            self.input_angles
        )

        maximum = max(
            self.input_angles
        )

        if (
            input_angle < minimum - eps
            or
            input_angle > maximum + eps
        ):

            raise ValueError(
                "input angle outside curve range"
            )

        input_angle = min(
            max(input_angle, minimum),
            maximum,
        )

        for index in range(
            len(self.input_angles) - 1
        ):

            x0 = self.input_angles[index]

            x1 = self.input_angles[index + 1]

            lower = min(
                x0,
                x1,
            )

            upper = max(
                x0,
                x1,
            )

            if lower <= input_angle <= upper:

                y0 = self.output_angles[index]

                y1 = self.output_angles[index + 1]

                factor = (
                    input_angle - x0
                ) / (
                    x1 - x0
                )

                return (
                    y0
                    +
                    factor
                    *
                    (
                        y1
                        -
                        y0
                    )
                )

        raise RuntimeError(
            "interpolation failed"
        )