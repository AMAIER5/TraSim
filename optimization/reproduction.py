"""
optimization/reproduction.py

Creation of new optimization candidates.

Issue #10: Reproduction previously used deterministic
round-robin (index % len(population)) to pick parents.
Combined with elitism-free evolution and mutation-only
(no crossover), this causes loss of diversity: the same
few survivors are cycled in the same order every
generation, and the population can collapse.

Fix: parents are now chosen randomly (with replacement)
from the survivor population.  This preserves diversity
and avoids the deterministic collapse.  A random
generator is accepted so the behaviour is reproducible
in tests.
"""

from __future__ import annotations

import random

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

    Parents are selected randomly (with replacement) from
    the survivor population.  Each child is a mutated copy
    of its parent.
    """

    def __init__(
        self,
        *,
        mutation: ParameterMutation,
        random_generator: random.Random | None = None,
    ):

        self.mutation = mutation

        self.random = (
            random_generator
            if random_generator is not None
            else random.Random()
        )

    def create(
        self,
        population: Population,
        *,
        count: int,
    ) -> Population:
        """
        Create mutated children.

        Parents are chosen randomly from the population.
        Each child is a mutated copy of its parent.

        Parameters
        ----------
        population:
            Survivor population to draw parents from.

        count:
            Number of children to create.

        Returns
        -------
        Population
            New population of mutated children.
        """

        if count <= 0:

            raise ValueError(
                "count must be positive"
            )

        children: list[ParameterSet] = []

        for _ in range(count):

            parent = self.random.choice(
                population.members
            )

            child = self.mutation.apply(
                parent
            )

            children.append(
                child
            )

        return Population(
            tuple(children)
        )