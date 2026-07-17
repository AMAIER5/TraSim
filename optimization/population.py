"""
optimization/population.py

Collection of optimization candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

from optimization.parameter_set import (
    ParameterSet,
)


@dataclass(frozen=True, slots=True)
class Population:
    """
    Immutable collection of candidate designs.

    One population represents one generation.
    """

    members: tuple[ParameterSet, ...]

    def __post_init__(self) -> None:

        if len(
            self.members
        ) == 0:

            raise ValueError(
                "population must not be empty"
            )

    def __len__(
        self,
    ) -> int:

        return len(
            self.members
        )

    def __getitem__(
        self,
        index: int,
    ) -> ParameterSet:

        return self.members[index]

    def __iter__(
        self,
    ):

        return iter(
            self.members
        )