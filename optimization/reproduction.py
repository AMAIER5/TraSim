"""
optimization/reproduction.py

Creation of new optimization candidates.
"""

from __future__ import annotations

from optimization.parameter_mutation import (
    ParameterMutation,
)

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.population import (
    Population,
)


class Reproduction:
    """
    Creates children from existing candidates.
    """

    def __init__(
        self,
        *,
        mutation: ParameterMutation,
    ):

        self.mutation = mutation

    def create(
        self,
        population: Population,
        *,
        count: int,
    ) -> Population:
        """
        Create mutated children.
        """

        if count <= 0:

            raise ValueError(
                "count must be positive"
            )

        children: list[ParameterSet] = []

        index = 0

        while len(children) < count:

            parent = population[
                index % len(population)
            ]

            child = self.mutation.apply(
                parent
            )

            children.append(
                child
            )

            index += 1

        return Population(
            tuple(children)
        )