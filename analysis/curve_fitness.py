"""
analysis/curve_fitness.py
"""

from __future__ import annotations

from analysis.error_metric import ErrorMetric
from analysis.target_curve import TargetCurve
from analysis.transfer_curve import TransferCurve

from optimization.fitness_function import FitnessFunction

from simulation.simulation_result import (
    SimulationResult,
)


class CurveFitness(FitnessFunction):
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

    def evaluate(
        self,
        simulation: tuple[SimulationResult, ...],
    ) -> float:
        """
        Evaluate a simulated mechanism.
        """

        if not simulation:
            raise ValueError("Simulation must contain at least one stage.")

        # For now, optimization uses the transfer curve of the
        # first stage.
        result = simulation[0]

        transfer_curve = TransferCurve(
            input_angles=result.input_angles,
            output_angles=result.output_angles,
        )

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

    def __call__(
        self,
        transfer_curve: TransferCurve,
    ) -> float:
        """
        Backwards-compatible interface.
        """

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