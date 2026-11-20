"""mechanics/reference_restore.py

Branch-faithful restoration of a driven lever's reference
output angle for a stage with a FROZEN coupling-rod length.

The reference output angle is restored on the correct
kinematic branch: a fine grid scan over the admissible
output angle range collects *all* solutions of the
rod-length closure, they are sorted by distance to the
reference angle, and the closest candidate is refined with
Newton iterations.  A plain AngleSolver/predict_output
approach would jump to the wrong branch.

This machinery is shared by the quasi-static builders
(mixed brake lever, pivot shift) so both approaches restore
states identically.
"""

from __future__ import annotations

import math

from mechanics.lever import Lever
from solver.constraints import rod_length_error

__all__ = [
    "NEWTON_MAX_ITERATIONS",
    "NEWTON_TOLERANCE",
    "RESTORE_TOLERANCE",
    "SEARCH_STEP",
    "MechanismBuildError",
    "restore_output_angle",
]


class MechanismBuildError(Exception):
    """Raised when a mechanism cannot be built for the
    requested brake/shift position.

    Typical causes are unreachable geometry (rod too short or
    too long for the moved pivot) or reference angles outside
    the admissible lever-angle segments.
    """


SEARCH_STEP = math.radians(0.05)
NEWTON_TOLERANCE = 1.0e-13
RESTORE_TOLERANCE = 1.0e-9
NEWTON_MAX_ITERATIONS = 60


def restore_output_angle(
    *,
    input_lever: Lever,
    output_lever: Lever,
    input_angle: float,
    rod_length: float,
    reference_angle: float,
    output_angle_min: float,
    output_angle_max: float,
) -> float:
    """Return the output lever angle closest to ``reference_angle``
    that closes the frozen rod between the input and output
    lever endpoints.

    When the reference angle already closes the rod (the
    unbraked state) it is returned unchanged.  Raises
    ``MechanismBuildError`` when no solution exists within the
    output angle bounds.
    """
    if not math.isfinite(
        output_angle_min,
    ) or not math.isfinite(
        output_angle_max,
    ):
        output_angle_min = -math.pi
        output_angle_max = math.pi
    if output_angle_max <= output_angle_min:
        raise MechanismBuildError(
            "empty output angle range"
        )

    input_point = input_lever.end_position(
        input_angle,
    )

    def residual(angle: float) -> float:
        return rod_length_error(
            input_point,
            output_lever.end_position(angle),
            rod_length,
        )

    if abs(residual(reference_angle)) <= (
        NEWTON_TOLERANCE
    ):
        return reference_angle

    candidates = _collect_branches(
        residual,
        output_angle_min,
        output_angle_max,
    )
    if not candidates:
        raise MechanismBuildError(
            "no kinematic solution for the moved "
            "pivot: rod cannot be connected within "
            "the output angle bounds"
        )
    candidates.sort(
        key=lambda angle: abs(
            angle - reference_angle
        )
    )
    angle = _newton_refine(
        residual,
        candidates[0],
        output_angle_min,
        output_angle_max,
    )
    if angle is None:
        raise MechanismBuildError(
            "reference output angle could not be "
            "restored within tolerance"
        )
    return angle


def _collect_branches(
    residual,
    minimum: float,
    maximum: float,
) -> list[float]:
    """Collect one start point per sign change of the rod
    closure residual over a fine angle grid."""
    step = SEARCH_STEP
    count = math.ceil((maximum - minimum) / step)
    candidates: list[float] = []
    previous_angle = minimum
    previous_value = residual(minimum)
    if previous_value == 0.0:
        candidates.append(minimum)
    for index in range(1, count + 1):
        angle = min(
            minimum + index * step,
            maximum,
        )
        value = residual(angle)
        if value == 0.0:
            candidates.append(angle)
        elif previous_value * value < 0.0:
            candidates.append(
                (previous_angle + angle) / 2.0,
            )
        previous_angle = angle
        previous_value = value
    return candidates


def _newton_refine(
    residual,
    start: float,
    minimum: float,
    maximum: float,
) -> float | None:
    """Newton refinement of a branch candidate; ``None``
    when the iteration leaves the bounds or the residual
    stays above the restore tolerance."""
    angle = start
    for _ in range(NEWTON_MAX_ITERATIONS):
        value = residual(angle)
        if abs(value) <= NEWTON_TOLERANCE:
            return angle
        h = 1.0e-7
        slope = (
            (residual(angle + h) - value) / h
        )
        if slope == 0.0:
            return None
        angle = angle - value / slope
        if angle < minimum or angle > maximum:
            return None
    if abs(residual(angle)) <= RESTORE_TOLERANCE:
        return angle
    return None
