"""
optimization/population.py

Collection of optimization candidates.

Issue #21: Added return type annotation to ``__iter__``.
``get_validation_results`` is not present in this module;
the issue may refer to a future or refactored version.
The ``__iter__`` annotation is the concrete fix here.
"""

from __future__ import annotations

from collections.abc import Iterator
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
    ) -> Iterator[ParameterSet]:
        """
        Issue #21: Added the return type annotation.

        Yields
        ------
        Iterator[ParameterSet]
            An iterator over the members of this population.
        """

        return iter(
            self.members
        )