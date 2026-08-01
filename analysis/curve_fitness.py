"""
analysis/curve_fitness.py
"""

from __future__ import annotations
from unittest import result

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

    Lower values are better.

    Failed simulations receive a penalty based on:
    - where the simulation stopped
    - how many points were successfully calculated

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
            raise ValueError(
                "Simulation must contain at least one stage."
            )

        # Currently only first stage is optimized.
        result = simulation[0]

        #   -------------------------------------------------
        #  Temp change start
        #   -------------------------------------------------

        print(
            "SIM:",
            "success=", result.success,
            "points=", len(result.input_angles),
            "blocked=", result.blocked_at,
        )

        #   -------------------------------------------------
        #  Temp change stop
        #   -------------------------------------------------
        
        # -------------------------------------------------
        # Invalid / blocked simulation
        # -------------------------------------------------

        if not result.success:

            calculated_points = len(
                result.input_angles
            )

            if result.blocked_at is not None:

                blocked_penalty = abs(
                    result.blocked_at
                )

            else:

                blocked_penalty = 100.0


            missing_penalty = (
                100.0
                *
                max(
                    0,
                    11 - calculated_points,
                )
            )

            return (
                1000.0
                +
                blocked_penalty
                +
                missing_penalty*10
            )


        # -------------------------------------------------
        # Valid simulation
        # -------------------------------------------------

        if len(result.input_angles) < 2:

            return 1e6


        transfer_curve = TransferCurve(
            input_angles=result.input_angles,
            output_angles=result.output_angles,
        )

        key = transfer_curve.input_angles

        metric = self._cache.get(
            key
        )

        if metric is None:

            target = self._target_curve.sample(
                key
            )

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

        metric = self._cache.get(
            key
        )

        if metric is None:

            target = self._target_curve.sample(
                key
            )

            metric = ErrorMetric(
                target=target,
            )

            self._cache[key] = metric


        return metric.calculate(
            transfer_curve,
        )