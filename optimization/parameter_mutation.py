"""
optimization/parameter_mutation.py

Mutation operator for optimization parameters.
"""

from __future__ import annotations

import random

from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)


class ParameterMutation:
    """
    Creates modified parameter sets.

    Mutation uses a relative change based
    on the parameter range.
    """

    def __init__(
        self,
        *,
        strength: float = 0.1,
        random_generator: random.Random | None = None,
    ):

        if strength < 0.0:

            raise ValueError(
                "strength must be positive"
            )

        self.strength = strength

        self.random = (
            random_generator
            if random_generator is not None
            else random.Random()
        )

    def apply(
        self,
        parameter_set: ParameterSet,
    ) -> ParameterSet:
        """
        Create mutated copy.
        """

        parameters = []

        for parameter in parameter_set.parameters:

            value = self._mutate_value(
                parameter
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

    def _mutate_value(
        self,
        parameter: Parameter,
    ) -> float:
        """
        Mutate single value.
        """

        if self.strength == 0.0:

            return parameter.value

        range_size = (
            parameter.maximum
            -
            parameter.minimum
        )

        delta = (
            self.random.uniform(
                -1.0,
                1.0,
            )
            *
            range_size
            *
            self.strength
        )

        value = (
            parameter.value
            +
            delta
        )

        return max(
            parameter.minimum,
            min(
                parameter.maximum,
                value,
            ),
        )