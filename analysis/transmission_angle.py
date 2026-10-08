"""analysis/transmission_angle.py

Soft minimum transmission-angle criterion for mixed brake
mechanisms.

The transmission angle ``mu`` of a stage is the angle between
the coupling rod and the output lever at the rod's attachment
point.  Small values indicate a near-singular stage: torque
transmission degrades and the kinematic behaviour becomes
ill-conditioned.  Instead of a hard rejection, the deviation of
the observed minimum transmission angle over the full brake
travel below a desired minimum is penalised softly:

    penalty = w * max(0, mu_min_desired - mu_min_observed)^2

A quadratic penalty keeps solutions with slightly reduced
transmission angles competitive while strongly discouraging
near-singular geometry.

Angles are in RADIANS.
"""

from __future__ import annotations

import math

from mechanics.mechanism import Mechanism
from mechanics.stage import Stage

__all__ = [
    "MIN_TRANSMISSION_ANGLE",
    "TransmissionAngleConstraint",
    "min_transmission_angle",
    "stage_transmission_angle",
]

MIN_TRANSMISSION_ANGLE = 0.35


def stage_transmission_angle(
    stage: Stage,
) -> float:
    """Transmission angle of one stage.

    The angle between the coupling rod and the output lever,
    measured at the rod's attachment point on the output lever:

        mu = arccos(
            (rod . output_lever_direction) / |rod|
        )

    with ``rod`` pointing from the output endpoint to the
    input endpoint.  The angle is normalized to the range
    [0, pi/2]: the supplement of an obtuse angle carries
    the same information (90 deg is optimal, small values
    are near-singular), so ``mu = min(mu, pi - mu)``.
    """
    input_end = stage.input_endpoint
    output_end = stage.output_endpoint
    rod = (
        input_end - output_end
    ).normalized()
    output_lever = (
        output_end - stage.output_lever.pivot
    ).normalized()
    cosine = rod.dot(output_lever)
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.acos(cosine)
    return min(angle, math.pi - angle)


def min_transmission_angle(
    mechanism: Mechanism,
) -> float:
    """Minimum transmission angle over all stages."""
    return min(
        stage_transmission_angle(stage)
        for stage in mechanism.stages
    )


class TransmissionAngleConstraint:
    """Soft penalty on the minimum transmission angle."""

    def __init__(
        self,
        minimum_angle: float = MIN_TRANSMISSION_ANGLE,
        weight: float = 1.0,
    ) -> None:
        """Parameters
        ----------
        minimum_angle:
            Desired minimum transmission angle mu_min
            (radians).  Typical design guidance is 40 deg
            for the worst case; the default of ~20 deg
            keeps the criterion soft.
        weight:
            Scaling factor of the quadratic penalty.
        """
        if minimum_angle < 0.0:
            raise ValueError(
                "minimum_angle must be non-negative."
            )
        if weight < 0.0:
            raise ValueError(
                "weight must be non-negative."
            )
        self._minimum_angle = minimum_angle
        self._weight = weight

    @property
    def minimum_angle(self) -> float:
        return self._minimum_angle

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(self, mechanisms: list[Mechanism]) -> float:
        """Penalty for a list of mechanisms (one per brake
        position).  The deviation of the observed minimum
        transmission angle below ``minimum_angle`` is squared
        and weighted; blocked candidates (no mechanisms)
        contribute nothing (hard penalties handle them).
        """
        if not mechanisms:
            return 0.0
        observed = min(
            min_transmission_angle(mechanism)
            for mechanism in mechanisms
        )
        deviation = max(
            0.0,
            self._minimum_angle - observed,
        )
        return self._weight * deviation * deviation
