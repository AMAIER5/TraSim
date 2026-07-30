"""
analysis/target_curve.py

Definition of desired kinematic behavior.
"""

from __future__ import annotations

from math import radians
import csv
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from analysis.transfer_curve import (
    TransferCurve,
)


@dataclass(frozen=True, slots=True)
class TargetCurve:
    """
    Mathematical definition of a desired
    input/output relationship.

    The function receives an input angle
    and returns the desired output angle.
    """

    function: Callable[[float], float]

    def __post_init__(self) -> None:

        if not callable(self.function):

            raise TypeError(
                "function must be callable"
            )

    @classmethod
    def from_points(
        cls,
        input_angles: tuple[float, ...],
        output_angles: tuple[float, ...],
    ) -> TargetCurve:
        """
        Create a target curve from sampled points.

        Linear interpolation is used between
        neighbouring points.
        """

        if len(input_angles) != len(
            output_angles
        ):
            raise ValueError(
                "input_angles and output_angles "
                "must have the same length."
            )

        if len(input_angles) < 2:
            raise ValueError(
                "At least two points are required."
            )

        if tuple(sorted(input_angles)) != input_angles:
            raise ValueError(
                "input_angles must be sorted."
            )

        def interpolate(
            angle: float,
        ) -> float:

            if angle <= input_angles[0]:
                return output_angles[0]

            if angle >= input_angles[-1]:
                return output_angles[-1]

            index = bisect_left(
                input_angles,
                angle,
            )

            x0 = input_angles[index - 1]
            x1 = input_angles[index]

            y0 = output_angles[index - 1]
            y1 = output_angles[index]

            t = (angle - x0) / (
                x1 - x0
            )

            return y0 + t * (
                y1 - y0
            )

        return cls(interpolate)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
    ) -> TargetCurve:
        """
        Load a target curve from CSV.

        Expected format:

        input_angle,output_angle
        0,0
        10,8
        20,17
        ...
        """

        input_angles: list[float] = []
        output_angles: list[float] = []

        with open(
            path,
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            required = {
                "input_angle",
                "output_angle",
            }

            if (
                reader.fieldnames is None
                or not required.issubset(
                    reader.fieldnames
                )
            ):
                raise ValueError(
                    "CSV must contain columns "
                    "'input_angle' and "
                    "'output_angle'."
                )

            for row in reader:
                input_angles.append(
                    radians(
                        float(row["input_angle"])
                    )
                )

                output_angles.append(
                    radians(
                        float(row["output_angle"])
                    )
                )

            return cls.from_points(
                tuple(input_angles),
                tuple(output_angles),
            )

    def evaluate(
        self,
        input_angle: float,
    ) -> float:
        """
        Evaluate target function.
        """

        return self.function(
            input_angle
        )

    def sample(
        self,
        input_angles: tuple[float, ...],
    ) -> TransferCurve:
        """
        Generate a transfer curve
        from sampled input values.
        """

        output_angles = tuple(
            self.evaluate(angle)
            for angle in input_angles
        )

        return TransferCurve(
            input_angles=input_angles,
            output_angles=output_angles,
        )