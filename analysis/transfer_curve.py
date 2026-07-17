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
        """

        if (
            input_angle
            <
            self.input_angles[0]
            or
            input_angle
            >
            self.input_angles[-1]
        ):
            raise ValueError(
                "input angle outside curve range"
            )

        for index in range(
            len(self.input_angles) - 1
        ):

            x0 = self.input_angles[index]

            x1 = self.input_angles[index + 1]

            if x0 <= input_angle <= x1:

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
                    (y1 - y0)
                )

        raise RuntimeError(
            "interpolation failed"
        )