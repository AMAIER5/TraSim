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
        initial_spread: float = 0.1,
    ):

        self.random = (
            random_generator
            or random
        )

        self.initial_spread = initial_spread

    def create(
        self,
        template: ParameterSet,
        *,
        size: int,
    ) -> Population:
        """
        Create initial population around
        template values.
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
        Create one randomized candidate
        around the template value.
        """

        parameters = []

        for parameter in template.parameters:

            range_size = (
                parameter.maximum
                -
                parameter.minimum
            )

            delta = (
                self.random.uniform(
                    -self.initial_spread,
                    self.initial_spread,
                )
                *
                range_size
            )

            value = (
                parameter.value
                +
                delta
            )

            value = max(
                parameter.minimum,
                min(
                    parameter.maximum,
                    value,
                ),
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