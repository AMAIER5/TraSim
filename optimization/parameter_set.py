"""
optimization/parameter_set.py

Collection of optimization parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from optimization.parameter import (
    Parameter,
)


@dataclass(frozen=True, slots=True)
class ParameterSet:
    """
    Immutable collection of parameters.

    Represents one possible design variant.
    """

    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:

        names = [
            parameter.name
            for parameter in self.parameters
        ]

        if len(names) != len(set(names)):

            raise ValueError(
                "duplicate parameter names"
            )

    def get(
        self,
        name: str,
    ) -> Parameter:
        """
        Return parameter by name.
        """

        for parameter in self.parameters:

            if parameter.name == name:

                return parameter

        raise KeyError(
            name
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.parameters
        )

    def values(
        self,
    ) -> dict[str, float]:
        """
        Return plain parameter values.
        """

        return {
            parameter.name: parameter.value
            for parameter in self.parameters
        }