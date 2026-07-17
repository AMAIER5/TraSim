"""
optimization/parameter.py

Definition of a single optimization parameter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Parameter:
    """
    A bounded optimization parameter.

    The optimizer may vary the value only
    inside the defined range.
    """

    name: str

    minimum: float

    maximum: float

    value: float

    def __post_init__(self) -> None:

        if self.minimum >= self.maximum:

            raise ValueError(
                "minimum must be smaller than maximum"
            )

        if not (
            self.minimum
            <=
            self.value
            <=
            self.maximum
        ):

            raise ValueError(
                "value outside parameter range"
            )

        if not self.name:

            raise ValueError(
                "parameter name must not be empty"
            )