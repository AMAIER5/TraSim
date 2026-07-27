"""
analysis/target_curve.py

Definition of desired kinematic behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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

        if not callable(
            self.function
        ):

            raise TypeError(
                "function must be callable"
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