"""
optimization/population_factory.py

Creates initial populations from
parameter templates.
"""

from __future__ import annotations

import random

from optimization.parameter import Parameter
from optimization.parameter_set import ParameterSet
from optimization.population import Population


class PopulationFactory:
    """
    Creates initial candidate populations.
    """

    def __init__(
        self,
        *,
        random_generator=None,
    ):

        self.random = (
            random_generator
            or random
        )

    def create(
        self,
        template: ParameterSet,
        *,
        size: int,
    ) -> Population:
        """
        Create random population.
        """

        if size <= 0:
            raise ValueError(
                "size must be positive"
            )

        candidates = []

        for _ in range(size):

            candidates.append(
                self._create_candidate(
                    template
                )
            )

        return Population(
            tuple(candidates)
        )

    def _create_candidate(
        self,
        template: ParameterSet,
    ) -> ParameterSet:
        """
        Create one randomized candidate.
        """

        parameters = []

        for parameter in template.parameters:

            value = self.random.uniform(
                parameter.minimum,
                parameter.maximum,
            )

            parameters.append(
                Parameter(
                    name=parameter.name,
                    minimum=parameter.minimum,
                    maximum=parameter.maximum,
                    value=value,
                )
            )

        return ParameterSet(
            tuple(parameters)
        )