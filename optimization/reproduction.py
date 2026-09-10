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

Optional crossover: when a ``crossover`` operator is
provided, each child is produced by recombining two
randomly chosen parents and then mutating the result.
This lets the optimiser assemble good sub-solutions
(for example a good lever 1/2 configuration from one
parent with a good lever 3/4 configuration from another)
instead of relying on single-parent mutation alone.
When ``crossover`` is ``None`` the original mutations-
only behaviour is preserved.
"""

from __future__ import annotations

import random

from optimization.crossover import (
    Crossover,
)
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
    of its parent.  When a crossover operator is provided,
    each child is instead produced by recombining two
    parents and then mutating the result.
    """

    def __init__(
        self,
        *,
        mutation: ParameterMutation,
        random_generator: random.Random | None = None,
        crossover: Crossover | None = None,
    ) -> None:

        self.mutation = mutation

        self.random = (
            random_generator
            if random_generator is not None
            else random.Random()
        )

        self.crossover = crossover

    def create(
        self,
        population: Population,
        *,
        count: int,
    ) -> Population:
        """
        Create mutated children.

        Parents are chosen randomly from the population.
        When crossover is enabled each child is the
        recombination of two parents followed by mutation;
        otherwise each child is a mutated copy of a single
        parent.

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

            child = self._create_child(
                population
            )

            children.append(
                child
            )

        return Population(
            tuple(children)
        )

    def _create_child(
        self,
        population: Population,
    ) -> ParameterSet:
        """
        Create a single child.

        With crossover: recombine two randomly chosen
        parents and mutate the result.  Without crossover:
        mutate a single randomly chosen parent.
        """

        if self.crossover is None:

            parent = self.random.choice(
                population.members
            )

            return self.mutation.apply(
                parent
            )

        parent_a = self.random.choice(
            population.members
        )

        parent_b = self.random.choice(
            population.members
        )

        recombined = self.crossover.recombine(
            parent_a,
            parent_b,
        )

        return self.mutation.apply(
            recombined
        )