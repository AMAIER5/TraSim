"""
analysis/endpoint_constraint.py

Endpoint constraint for braked transfer curves.

Purpose: prevent the observed branch jump of the braked curve by
penalising the squared deviation of the output angle at the maximum
brake position (outermost position) from a prescribed end angle.
The term is additive and quadratic, so small deviations are tolerated
while large branch jumps dominate the fitness.
"""
from __future__ import annotations


class EndpointConstraint:
    """
    Quadratic penalty on the endpoint deviation.

    penalty = (phi_out(beta_max, outermost position) - phi_end) ** 2
    """

    def __init__(
        self,
        end_angle: float,
        weight: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        end_angle:
            Desired output angle phi_end at the outermost position.
        weight:
            Scaling factor of the squared deviation.
        """
        self._end_angle = end_angle
        self._weight = weight

    @property
    def end_angle(self) -> float:
        return self._end_angle

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(
        self,
        output_angles: tuple[float, ...],
    ) -> float:
        """
        Squared deviation of the last output angle from the end angle.

        The outermost position corresponds to the last simulated
        output angle.  An empty curve cannot deviate, so the penalty
        is zero.
        """
        if not output_angles:
            return 0.0
        deviation = output_angles[-1] - self._end_angle
        return self._weight * deviation * deviation
