"""
mechanics/mechanism.py

Container for multiple connected mechanism stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from mechanics.stage import Stage


@dataclass(frozen=True, slots=True)
class Mechanism:
    """
    Collection of connected stages.

    The order of stages defines the kinematic chain.
    """

    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:

        if not isinstance(
            self.stages,
            tuple,
        ):
            raise TypeError(
                "stages must be a tuple"
            )