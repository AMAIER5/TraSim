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


# Tolerance for matching a queried input angle against an exact
# support point of a discrete target curve.  Small floating-point
# differences (e.g. from math.radians applied at different call
# sites) must be tolerated so that simulation angles generated
# from the same CSV support points are recognised as matches.
_DISCRETE_ANGLE_TOLERANCE = 1e-9


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

        input_angles, output_angles = _read_csv_points(
            path,
        )

        return cls.from_points(
            tuple(input_angles),
            tuple(output_angles),
        )

    @classmethod
    def from_points_strict(
        cls,
        input_angles: tuple[float, ...],
        output_angles: tuple[float, ...],
    ) -> DiscreteTargetCurve:
        """
        Create a non-interpolating target curve from sampled points.

        Unlike ``from_points`` (which linearly interpolates
        between neighbouring points), a strict curve is only
        defined at the given support points.  Evaluating or
        sampling at an input angle that does not coincide with a
        support point raises a ``ValueError``.  This guarantees
        that the fitness is computed exclusively at the
        prescribed support points and never uses interpolated
        translations between them.
        """

        return DiscreteTargetCurve(
            input_angles=input_angles,
            output_angles=output_angles,
        )

    @classmethod
    def from_csv_strict(
        cls,
        path: str | Path,
    ) -> DiscreteTargetCurve:
        """
        Load a non-interpolating target curve from CSV.

        Same CSV format as ``from_csv``, but the resulting curve
        is only defined at the support points read from the file
        (no interpolation between them).
        """

        input_angles, output_angles = _read_csv_points(
            path,
        )

        return cls.from_points_strict(
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


class DiscreteTargetCurve:
    """
    Non-interpolating target curve.

    Unlike ``TargetCurve``, which is defined by an arbitrary
    function and therefore defined everywhere, a
    ``DiscreteTargetCurve`` is only defined at a fixed set of
    support points (input_angle, output_angle).  Evaluating or
    sampling at an input angle that is not (within tolerance) a
    support point raises a ``ValueError``.

    This is the strict counterpart to ``TargetCurve.from_points``:
    it guarantees that fitness evaluation only happens at the
    prescribed support points and never uses linear interpolation
    between them.  ``sample`` returns a ``TransferCurve`` only
    when every requested input angle coincides with a support
    point, so the fitness comparison is performed exclusively at
    the support points.
    """

    def __init__(
        self,
        *,
        input_angles: tuple[float, ...],
        output_angles: tuple[float, ...],
    ) -> None:

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

        self._input_angles = input_angles
        self._output_angles = output_angles

    @property
    def input_angles(self) -> tuple[float, ...]:
        """Support point input angles."""

        return self._input_angles

    @property
    def output_angles(self) -> tuple[float, ...]:
        """Support point output angles."""

        return self._output_angles

    def evaluate(
        self,
        input_angle: float,
    ) -> float:
        """
        Evaluate the target at an input angle.

        Raises a ``ValueError`` if the angle does not coincide
        (within ``_DISCRETE_ANGLE_TOLERANCE``) with a support
        point, so the curve is never evaluated by interpolation.
        """

        index = self._support_index(
            input_angle,
        )

        return self._output_angles[index]

    def sample(
        self,
        input_angles: tuple[float, ...],
    ) -> TransferCurve:
        """
        Generate a transfer curve from sampled input values.

        Every requested input angle must coincide (within
        tolerance) with a support point; otherwise a
        ``ValueError`` is raised.  This prevents the fitness from
        evaluating interpolated values between support points.
        """

        output_angles = tuple(
            self._output_angles[
                self._support_index(angle)
            ]
            for angle in input_angles
        )

        return TransferCurve(
            input_angles=input_angles,
            output_angles=output_angles,
        )

    def _support_index(
        self,
        input_angle: float,
    ) -> int:
        """
        Return the index of the support point matching the
        given input angle, or raise a ``ValueError``.
        """

        index = self.support_index(input_angle)

        if index is None:
            raise ValueError(
                f"input angle {input_angle} is not a "
                "support point of the discrete target curve"
            )

        return index

    def support_index(
        self,
        input_angle: float,
    ) -> int | None:
        """
        Return the index of the support point matching the
        given input angle (within tolerance), or ``None`` if the
        angle is not a support point.

        This allows callers to filter simulated input angles down
        to the prescribed support points without interpolating
        between them.
        """

        for index, support in enumerate(
            self._input_angles
        ):

            if abs(
                input_angle - support
            ) <= _DISCRETE_ANGLE_TOLERANCE:
                return index

        return None


def _read_csv_points(
    path: str | Path,
) -> tuple[list[float], list[float]]:
    """
    Read input/output angle pairs (in degrees) from a CSV file
    and return them converted to radians.
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

        return input_angles, output_angles