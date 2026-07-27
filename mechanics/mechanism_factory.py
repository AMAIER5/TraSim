"""
mechanics/mechanism_factory.py

Factory for creating mechanisms from
optimization parameters.
"""

from __future__ import annotations

from collections.abc import Callable

from optimization.parameter_set import (
    ParameterSet,
)


class MechanismFactory:
    """
    Creates mechanisms from parameter sets.

    The factory translates optimization
    parameters into mechanical objects.
    """

    def __init__(
        self,
        *,
        builder: Callable[
            [ParameterSet],
            object,
        ],
    ):
        """
        Create factory.

        builder:
            Function responsible for
            constructing the mechanism.
        """

        if not callable(builder):

            raise TypeError(
                "builder must be callable"
            )

        self.builder = builder

    def create(
        self,
        parameters: ParameterSet,
    ) -> object:
        """
        Create mechanism from parameters.
        """

        return self.builder(
            parameters
        )