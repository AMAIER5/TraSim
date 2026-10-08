"""
analysis/brake_curve_fitness.py

Fitness over multiple brake positions.

A candidate is evaluated over ALL (brake position, target curve)
pairs simultaneously:

    fitness = sum_k w_k * error_k
              + w_max * max_k error_k
              + blocking penalties
              + endpoint constraint term

The per-position errors use the established ErrorMetric/TransferCurve
machinery from analysis.error_metric.  The max term prevents the
optimizer from perfectly matching one brake position while discarding
another.

Blocking: if ANY (brake position, target curve) pair blocks or a
MechanismBuildError occurs, the whole candidate receives a hard
penalty.  The penalty logic reuses the constants from
analysis.curve_fitness; PENALTY_MAX_BLOCKING is re-exported here.

Endpoint constraint: the squared deviation of the output angle at the
maximum brake position (outermost position) from a prescribed end
angle is added, preventing the observed branch jump of the braked
curve.
"""
from __future__ import annotations

import logging

from analysis.curve_fitness import (
    PENALTY_BASE,
    PENALTY_INSUFFICIENT_POINTS,
    PENALTY_MAX_BLOCKING,
)
from analysis.endpoint_constraint import EndpointConstraint
from analysis.error_metric import ErrorMetric
from analysis.target_curve import TargetCurve
from analysis.transfer_curve import TransferCurve
from simulation.simulation_result import SimulationResult

logger = logging.getLogger(__name__)

__all__ = [
    "PENALTY_MAX_BLOCKING",
    "BrakeCurveFitness",
]


class BrakeCurveFitness:
    """
    Fitness across multiple (brake position, target curve) pairs.

    Each pair contributes a mean absolute error (via ErrorMetric)
    weighted by its per-position weight w_k.  The maximum error is
    added with weight w_max so no single brake position can be
    sacrificed.  Blocking in ANY pair yields a hard penalty for the
    entire candidate.
    """

    def __init__(
        self,
        *,
        weights: tuple[float, ...] | None = None,
        max_weight: float = 1.0,
        endpoint_constraint: EndpointConstraint | None = None,
    ) -> None:
        """
        Parameters
        ----------
        weights:
            Per brake position weights w_k.  Defaults to 1 for
            every pair.  Must match the number of pairs at
            evaluation time when given.
        max_weight:
            Weight w_max of the max-error term.
        endpoint_constraint:
            Optional EndpointConstraint evaluated on the output
            angles of the maximum brake position (the last pair).
        """
        self._weights = weights
        self._max_weight = max_weight
        self._endpoint_constraint = endpoint_constraint
        self._cache: dict[tuple[float, ...], ErrorMetric] = {}

    def _pair_errors(
        self,
        pairs: list[tuple[SimulationResult, TargetCurve]],
    ) -> list[float]:
        errors: list[float] = []
        for result, target in pairs:
            if len(result.input_angles) < 2:
                errors.append(PENALTY_INSUFFICIENT_POINTS)
                continue
            transfer_curve = TransferCurve(
                input_angles=result.input_angles,
                output_angles=result.output_angles,
            )
            key = (id(target), transfer_curve.input_angles)
            if key not in self._cache:
                self._cache[key] = ErrorMetric(
                    target=target.sample(transfer_curve.input_angles)
                )
            errors.append(
                self._cache[key].calculate(transfer_curve)
            )
        return errors

    def evaluate(
        self,
        pairs: list[tuple[SimulationResult, TargetCurve]],
    ) -> float:
        """
        Evaluate a candidate over all brake positions.

        Parameters
        ----------
        pairs:
            One (SimulationResult, TargetCurve) pair per brake
            position, ordered by increasing brake position; the
            last pair is the maximum brake position (outermost).

        Returns
        -------
        float
            Fitness value (lower is better).  Blocking in any pair
            results in fitness >= PENALTY_BASE.
        """
        if not pairs:
            raise ValueError(
                "At least one (simulation, target) pair is required."
            )

        weights = self._weights
        if weights is None:
            weights = tuple(1.0 for _ in pairs)
        if len(weights) != len(pairs):
            raise ValueError(
                "Number of weights must match number of pairs."
            )

        if any(not result.success for result, _ in pairs):
            logger.debug("Blocking brake position detected.")
            return PENALTY_MAX_BLOCKING

        errors = self._pair_errors(pairs)
        weighted_sum = sum(
            weight * error for weight, error in zip(weights, errors)
        )
        max_term = self._max_weight * max(errors)

        endpoint_term = 0.0
        if self._endpoint_constraint is not None:
            endpoint_term = self._endpoint_constraint.evaluate(
                pairs[-1][0].output_angles
            )

        fitness = weighted_sum + max_term + endpoint_term
        logger.debug("Brake curve fitness: %s", fitness)
        return fitness

    def __call__(
        self,
        pairs: list[tuple[SimulationResult, TargetCurve]],
    ) -> float:
        return self.evaluate(pairs)
