"""
mechanics/csv_mechanism_builder.py

Build a mechanical mechanism from a MechanismDefinition.
"""

from __future__ import annotations

from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from model.mechanism_definition import MechanismDefinition
from optimization.mechanism_builder import MechanismBuilder
from optimization.parameter_set import ParameterSet


class CsvMechanismBuilder(MechanismBuilder):
    """
    Build a Mechanism from a MechanismDefinition.

    The builder converts the abstract model definition
    into simulation-ready mechanical components.
    """

    def __init__(
        self,
        definition: MechanismDefinition,
    ) -> None:
        self._definition = definition

    def build(
        self,
        parameters: ParameterSet,
    ) -> Mechanism:
        """
        Build a mechanism.

        Parameters are currently ignored. A later sprint
        will map optimization parameters onto the
        mechanism definition before constructing the
        mechanism.
        """

        levers = self._create_levers(
            self._definition,
        )

        stages = self._create_stages(
            self._definition,
            levers,
        )

        return Mechanism(
            stages=tuple(stages),
        )

    def _create_levers(
        self,
        definition: MechanismDefinition,
    ) -> dict[int, Lever]:
        """
        Create mechanical levers.
        """

        result: dict[int, Lever] = {}

        for lever_definition in definition.levers:
            result[lever_definition.id] = Lever(
                pivot=lever_definition.pivot,
                axis=lever_definition.axis,
                length=lever_definition.length_start,
            )

        return result

    def _create_stages(
        self,
        definition: MechanismDefinition,
        levers: dict[int, Lever],
    ) -> list[Stage]:
        """
        Create stages from driver relations.
        """

        stages: list[Stage] = []

        for lever_definition in definition.levers:
            if lever_definition.driver is None:
                continue

            stages.append(
                Stage.from_reference_position(
                    input_lever=levers[lever_definition.driver],
                    output_lever=levers[lever_definition.id],
                    input_angle=0.0,
                    output_angle=0.0,
                )
            )

        return stages