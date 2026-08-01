"""
simulation/motion_provider.py

Common interface for motion angle providers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class MotionProvider(Protocol):
    """
    Protocol for simulation input angle providers.

    Motion providers generate input angles and may receive
    simulation feedback after successful solver steps.
    """

    def __iter__(self) -> Iterator[float]:
        """
        Generate input angles.
        """
        ...

    def feedback(
        self,
        *,
        output_delta: float,
    ) -> None:
        """
        Receive output motion feedback.

        Non-adaptive providers may ignore this callback.
        """
        ...