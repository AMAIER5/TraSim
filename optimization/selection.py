"""
optimization/selection.py

Selection of best optimization candidates.
"""

from __future__ import annotations

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.population import (
    Population,
)


class Selection:
    """
    Selects candidates with lowest score.

    Lower fitness values are better.
    """

    def select(
        self,
        population: Population,
        scores: dict[
            ParameterSet,
            float,
        ],
        *,
        count: int,
    ) -> Population:
        """
        Return best candidates.
        """

        if count <= 0:

            raise ValueError(
                "count must be positive"
            )

        if count > len(population):

            raise ValueError(
                "cannot select more candidates than available"
            )

        ranked = sorted(
            population,
            key=lambda candidate:
                scores[candidate],
        )

        return Population(
            tuple(
                ranked[:count]
            )
        )