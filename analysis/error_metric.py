"""
analysis/error_metric.py

Error calculation between transfer curves.
"""

from __future__ import annotations

import math

from analysis.transfer_curve import (
    TransferCurve,
)


class ErrorMetric:
    """
    Calculates deviation from a target curve.

    Current implementation:
    Mean absolute error (MAE)
    over sampled input points.
    """

    def __init__(
        self,
        target: TransferCurve,
    ):

        self.target = target

    def calculate(
        self,
        actual: TransferCurve,
    ) -> float:
        """
        Calculate mean absolute angular error.
        """

        if (
            actual.input_angles
            !=
            self.target.input_angles
        ):
            raise ValueError(
                "input sampling must match"
            )

        errors = []

        for input_angle, target_output in zip(
            self.target.input_angles,
            self.target.output_angles,
        ):

            actual_output = actual.output_at(
                input_angle
            )

            errors.append(
                abs(
                    actual_output
                    -
                    target_output
                )
            )

        return (
            sum(errors)
            /
            len(errors)
        )