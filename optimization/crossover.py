"""
optimization/crossover.py

Recombination of two parent candidates.

Crossover combines the parameters of two parents into a
new child, which is then mutated by the ``Reproduction``
operator.  This lets the optimiser assemble good
sub-solutions (for example a good lever 1/2
configuration from one parent with a good lever 3/4
configuration from another) instead of relying on
single-parent mutation alone.

Two strategies are available:

- ``uniform``: each parameter is drawn independently from
  either parent with equal probability.
- ``grouped``: parameters are recombined in groups.  A
  group is the set of parameters that share the same
  ``lever.<id>.`` prefix, so all parameters of one lever
  are always inherited together from the same parent.
  This preserves intra-lever correlations (a lever's
  length and angle stay consistent) while still mixing
  complete levers between parents.
"""

from __future__ import annotations

import random

from optimization.parameter_set import (
    ParameterSet,
)

# Available crossover strategies.
UNIFORM = "uniform"
GROUPED = "grouped"

_VALID_STRATEGIES = (UNIFORM, GROUPED)


def _lever_key(parameter_name: str) -> str:
    """
    Return the group key for a parameter name.

    Parameters are named ``lever.<id>.<attr>``.  All
    parameters of one lever share the prefix
    ``lever.<id>.`` and are grouped together.  Any
    parameter that does not match this scheme forms its
    own singleton group (its full name), so it is
    inherited as a unit too.
    """

    parts = parameter_name.split(".")

    if len(parts) >= 2 and parts[0] == "lever":

        return ".".join(parts[:2])

    return parameter_name


class Crossover:
    """
    Recombines two parent parameter sets into one child.
    """

    def __init__(
        self,
        *,
        strategy: str = UNIFORM,
        random_generator: random.Random | None = None,
    ) -> None:

        if strategy not in _VALID_STRATEGIES:

            raise ValueError(
                f"strategy must be one of "
                f"{_VALID_STRATEGIES}, got {strategy!r}"
            )

        self.strategy = strategy

        self.random = (
            random_generator
            if random_generator is not None
            else random.Random()
        )

    def recombine(
        self,
        parent_a: ParameterSet,
        parent_b: ParameterSet,
    ) -> ParameterSet:
        """
        Create one child from two parents.

        Both parents must carry the same parameter names in
        the same order (this is guaranteed for candidates
        created from the same template by the population
        factory).
        """

        names_a = [
            parameter.name
            for parameter in parent_a.parameters
        ]

        names_b = [
            parameter.name
            for parameter in parent_b.parameters
        ]

        if names_a != names_b:

            raise ValueError(
                "parents must have identical parameter names"
            )

        if self.strategy == UNIFORM:

            return self._recombine_uniform(
                parent_a,
                parent_b,
            )

        return self._recombine_grouped(
            parent_a,
            parent_b,
        )

    def _recombine_uniform(
        self,
        parent_a: ParameterSet,
        parent_b: ParameterSet,
    ) -> ParameterSet:
        """
        Per-parameter uniform crossover: each parameter is
        taken from either parent with equal probability.
        """

        parameters = []

        for index, parameter in enumerate(
            parent_a.parameters
        ):

            if self.random.random() < 0.5:

                source = parameter

            else:

                source = parent_b.parameters[index]

            parameters.append(source)

        return ParameterSet(
            tuple(parameters)
        )

    def _recombine_grouped(
        self,
        parent_a: ParameterSet,
        parent_b: ParameterSet,
    ) -> ParameterSet:
        """
        Grouped crossover: parameters of one lever are
        always inherited together from the same parent.
        """

        # Map group key -> list of (index, parameter).
        groups: dict[str, list[int]] = {}

        for index, parameter in enumerate(
            parent_a.parameters
        ):

            key = _lever_key(parameter.name)

            groups.setdefault(
                key,
                [],
            ).append(index)

        # Decide the parent for each group.
        group_parent: dict[str, ParameterSet] = {}

        for key in groups:

            if self.random.random() < 0.5:

                group_parent[key] = parent_a

            else:

                group_parent[key] = parent_b

        parameters = [
            None
        ] * len(parent_a.parameters)

        for key, indices in groups.items():

            source = group_parent[key]

            for index in indices:

                parameters[index] = source.parameters[index]

        return ParameterSet(
            tuple(parameters)
        )
