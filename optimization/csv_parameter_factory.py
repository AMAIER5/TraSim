"""
optimization/csv_parameter_factory.py

Creates optimization parameter templates
from a CSV-based MechanismDefinition.
"""

from __future__ import annotations

from model.mechanism_definition import (
    MechanismDefinition,
)

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_set import (
    ParameterSet,
)


class CsvParameterFactory:
    """
    Creates optimization parameters from
    lever definitions.

    Each optimizable lever property is mapped
    to a Parameter.
    """

    @staticmethod
    def create(
        definition: MechanismDefinition,
    ) -> ParameterSet:
        """
        Create a parameter template from
        a mechanism definition.
        """

        parameters = []

        for lever in definition.levers:

            parameters.append(
                Parameter(
                    name=f"lever.{lever.id}.length",
                    minimum=lever.length_min,
                    maximum=lever.length_max,
                    value=lever.length_start,
                )
            )

        return ParameterSet(
            parameters=tuple(parameters),
        )