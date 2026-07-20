"""
optimization/curve_fitness.py
"""

from __future__ import annotations

from analysis.error_metric import ErrorMetric
from analysis.target_curve import TargetCurve
from analysis.transfer_curve import TransferCurve


class CurveFitness:
    """
    Calculates the fitness of a simulated transfer curve.

    ErrorMetric instances are cached for identical
    input-angle grids.
    """

    def __init__(
        self,
        *,
        target_curve: TargetCurve,
    ) -> None:

        self._target_curve = target_curve

        self._cache: dict[
            tuple[float, ...],
            ErrorMetric,
        ] = {}

    def __call__(
        self,
        transfer_curve: TransferCurve,
    ) -> float:

        key = transfer_curve.input_angles

        metric = self._cache.get(key)

        if metric is None:

            target = self._target_curve.sample(key)

            metric = ErrorMetric(
                target=target,
            )

            self._cache[key] = metric

        return metric.calculate(
            transfer_curve,
        )