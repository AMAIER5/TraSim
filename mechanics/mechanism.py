"""
mechanics/mechanism.py

Container for multiple connected mechanism stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from mechanics.lever import Lever
from mechanics.stage import Stage


@dataclass(frozen=True, slots=True)
class MechanismLever:
    """
    One lever of a mechanism together with the lever
    angle of its reference position.

    ``coupled_to`` is the one-based index (within the
    mechanism's lever list) of the lever this lever is
    torsion-coupled to, or ``None``.  The coupling fixes
    the angle difference between the two levers, so the
    angle of either lever can be derived from the other.
    """

    lever: Lever
    reference_angle: float
    coupled_to: int | None = None


@dataclass(frozen=True, slots=True)
class Mechanism:
    """
    Collection of connected stages.

    The order of stages defines the kinematic chain.

    ``levers`` optionally lists every lever of the
    mechanism (including levers that appear in no stage,
    for example a drive lever that is only torsion-coupled
    to the input lever of the first stage) in definition
    order.  Visualizations use it to draw the complete
    mechanism; when it is ``None`` the levers are derived
    from the stages.
    """

    stages: tuple[Stage, ...]
    levers: (
        tuple[MechanismLever, ...] | None
    ) = None

    def __post_init__(self) -> None:

        if not isinstance(
            self.stages,
            tuple,
        ):
            raise TypeError(
                "stages must be a tuple"
            )
