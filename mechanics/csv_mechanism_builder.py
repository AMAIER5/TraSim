"""
mechanics/csv_mechanism_builder.py

Build a mechanical mechanism from a MechanismDefinition.

Rod lengths are calculated automatically from lever positions at reference angles.
Each stage's rod length is computed as the Euclidean distance between the input and output
lever endpoints at their respective reference angles (angle_start from the CSV definition).

The rod length is computed once during stage construction using:
    rod_length = || output_lever.end_position(angle_start) - input_lever.end_position(angle_start) ||

This ensures the mechanism is valid at the reference configuration.
However, the mechanism may still block during motion if the geometry doesn't allow
the full angle range. For guaranteed motion, ensure:
1. The pivot distance >= |input_length - output_length|
2. The pivot distance <= input_length + output_length
3. The motion range is within the geometric feasibility limits
"""

from __future__ import annotations

from dataclasses import replace

from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from model.mechanism_definition import MechanismDefinition
from optimization.mechanism_builder import MechanismBuilder
from optimization.parameter_set import ParameterSet
from validation.stage_motion_validator import StageMotionValidator

class CsvMechanismBuilder(MechanismBuilder):
    """
    Build a Mechanism from a MechanismDefinition.

    The builder converts the abstract model definition
    into simulation-ready mechanical components.

    Rod length for each stage is automatically calculated
    as the Euclidean distance between the input and output
    lever endpoints at their reference angles (angle_start).
    """

    def __init__(
        self,
        definition: MechanismDefinition,
        *,
        validator: StageMotionValidator | None = None,
    ) -> None:

        self._definition = definition
        self._validator = (
            validator
            if validator is not None
            else StageMotionValidator()
        )
        self._validation_results = []

    def build(
        self,
        parameters: ParameterSet,
    ) -> Mechanism:
        """
        Build a mechanism from the CSV definition.

        Optimization parameters are applied as an overlay
        before creating the mechanical model.
        """

        definition = self._apply_parameters(
            self._definition,
            parameters,
        )

        levers = self._create_levers(definition)

        stages = self._create_stages(
            definition,
            levers,
        )

        self._validation_results = []

        for index, stage in enumerate(stages):

            result = self._validator.validate(
                stage,
                stage_id=index,
            )

            self._validation_results.append(result)

        return Mechanism(stages=tuple(stages))

    def _apply_parameters(
        self,
        definition: MechanismDefinition,
        parameters: ParameterSet,
    ) -> MechanismDefinition:
        """
        Apply optimization parameters to a mechanism definition.
        The original CSV definition remains unchanged.
        """

        values = parameters.values()
        levers = []

        for lever in definition.levers:

            length = values.get(
                f"lever.{lever.id}.length",
                lever.length_start,
            )

            angle = values.get(
                f"lever.{lever.id}.angle",
                lever.angle_start,
            )

            levers.append(
                replace(
                    lever,
                    length_start=length,
                    angle_start=angle,
                )
            )

        return MechanismDefinition(levers=tuple(levers))

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

        Each stage's rod length is automatically calculated as the
        distance between the input and output lever endpoints at
        their reference angles (angle_start from the definition).

        The rod length is computed as:
            rod_length = ||output_endpoint - input_endpoint||

        where endpoints are calculated at the reference angles:
            input_endpoint = input_lever.end_position(input_angle_start)
            output_endpoint = output_lever.end_position(output_angle_start)
        """

        stages: list[Stage] = []

        definitions = {
            lever.id: lever
            for lever in definition.levers
        }

        for lever_definition in definition.levers:

            if lever_definition.driver is None:
                continue

            driver_definition = definitions[lever_definition.driver]

            # Use the updated reference angles from parameters
            stages.append(
                Stage.from_reference_position(
                    input_lever=levers[lever_definition.driver],
                    output_lever=levers[lever_definition.id],

                    input_angle=driver_definition.angle_start,
                    output_angle=lever_definition.angle_start,

                    input_angle_min=driver_definition.angle_min,
                    input_angle_max=driver_definition.angle_max,

                    output_angle_min=lever_definition.angle_min,
                    output_angle_max=lever_definition.angle_max,

                    validate_reference=False,
                )
            )

        return stages

    def get_validation_results(self) -> tuple:
        """
        Return stage validation results.
        """
        return tuple(self._validation_results)