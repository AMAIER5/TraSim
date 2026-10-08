"""analysis/brake_grid.py

Brake grid with evenly spaced brake positions and derived
target curves.

The brake lever travels between ``brake_min`` and ``brake_max``
(unbraked reference at ``brake_min``).  Intermediate target
curves are generated as shifted derivatives of the two given
base curves: for a brake fraction ``f`` in [0, 1] the target is
the point-wise blend

    target_f(x) = (1 - f) * unbraked(x)
                        + f * braked(x)

evaluated on the common support grid of input angles.  Both
base curves share the endpoint characteristic (identical output
at the maximum input angle), so every blended derivative shares
it as well.

User I/O angles are in DEGREES, internal calculations use
RADIANS.
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.target_curve import TargetCurve

__all__ = [
    "BrakeGridSettings",
    "blend_target_curves",
    "make_brake_grid",
]


@dataclass(frozen=True, slots=True)
class BrakeGridSettings:
    """Configuration of the evenly spaced brake grid.

    ``brake_min_deg`` is the unbraked reference position (0 %
    brake travel), ``brake_max_deg`` the fully braked position
    (100 % brake travel).
    """

    brake_min_deg: float
    brake_max_deg: float
    position_count: int = 5

    def __post_init__(self) -> None:
        if self.position_count < 2:
            raise ValueError(
                "position_count must be at least 2 "
                "(unbraked and fully braked)."
            )
        if self.brake_max_deg < self.brake_min_deg:
            raise ValueError(
                "brake_max_deg must be greater than or "
                "equal to brake_min_deg."
            )

    @property
    def fractions(self) -> tuple[float, ...]:
        """Brake fractions of the grid positions (0..1)."""
        count = self.position_count
        if count == 2:
            return (0.0, 1.0)
        step = 1.0 / (count - 1)
        return tuple(
            index * step for index in range(count)
        )

    @property
    def brake_positions_deg(self) -> tuple[float, ...]:
        """Brake lever angles of the grid in degrees."""
        minimum = self.brake_min_deg
        span = self.brake_max_deg - self.brake_min_deg
        return tuple(
            minimum + fraction * span
            for fraction in self.fractions
        )


def make_brake_grid(
    settings: BrakeGridSettings,
) -> tuple[float, ...]:
    """Brake grid positions in radians, ordered by
    increasing brake travel."""
    from math import radians

    return tuple(
        radians(position)
        for position in settings.brake_positions_deg
    )


def blend_target_curves(
    unbraked: TargetCurve,
    braked: TargetCurve,
    fractions: tuple[float, ...],
    *,
    input_angles: tuple[float, ...],
) -> tuple[TargetCurve, ...]:
    """Blend the two base curves into one target per fraction.

    Parameters
    ----------
    unbraked:
        Target curve at 0 % brake travel.
    braked:
        Target curve at 100 % brake travel.
    fractions:
        Brake fractions of the requested grid positions.
        Values outside [0, 1] are rejected.
    input_angles:
        Common support grid (radians, sorted) the blended
        curves are sampled on.

    Returns
    -------
    tuple[TargetCurve, ...]
        One interpolating target curve per fraction, ordered
        like ``fractions``.
    """
    if len(input_angles) < 2:
        raise ValueError(
            "At least two input angles are required."
        )
    for fraction in fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "brake fractions must lie in [0, 1]."
            )
    targets = []
    for fraction in fractions:
        outputs = tuple(
            (1.0 - fraction)
            * unbraked.evaluate(angle)
            + fraction * braked.evaluate(angle)
            for angle in input_angles
        )
        targets.append(
            TargetCurve.from_points(
                input_angles=input_angles,
                output_angles=outputs,
            )
        )
    return tuple(targets)
