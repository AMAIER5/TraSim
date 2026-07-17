"""
analysis/fitness.py

Fitness evaluation for mechanism optimization.
"""

from __future__ import annotations

from analysis.error_metric import ErrorMetric
from analysis.target_curve import TargetCurve
from analysis.transfer_curve import TransferCurve


class Fitness:
    """
    Evaluates how well a mechanism matches
    a target behavior.

    Lower values indicate better results.
    """

    def __init__(
        self,
        *,
        target: TargetCurve,
        metric: ErrorMetric | None = None,
    ):

        self.target = target
        self.metric = metric

    def evaluate(
        self,
        curve: TransferCurve,
    ) -> float:
        """
        Calculate fitness value.
        """

        target_curve = self.target.sample(
            input_angles=curve.input_angles
        )

        if self.metric is None:

            metric = ErrorMetric(
                target_curve
            )

            return metric.calculate(
                curve
            )

        return self.metric.calculate(
            curve
        )