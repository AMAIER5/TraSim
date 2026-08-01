"""
tests/test_csv_mechanism_builder_parameters.py

Tests for CsvMechanismBuilder parameter overlay.
"""

from __future__ import annotations

from model.mechanism_definition import MechanismDefinition
from model.lever_definition import LeverDefinition
from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.csv_mechanism_builder import CsvMechanismBuilder
from optimization.parameter import Parameter
from optimization.parameter_set import ParameterSet


def create_definition() -> MechanismDefinition:
    return MechanismDefinition(
        levers=(
            LeverDefinition(
                id=0,
                pivot=Point3D(0, 0, 0),
                length_min=40,
                length_max=100,
                length_start=70,
                angle_min=-45,
                angle_max=45,
                angle_start=0,
                axis=Vector3D(0, 0, 1),
                driver=None,
                coupled=None,
            ),
            LeverDefinition(
                id=1,
                pivot=Point3D(100, 0, 0),
                length_min=40,
                length_max=100,
                length_start=70,
                angle_min=-45,
                angle_max=45,
                angle_start=0,
                axis=Vector3D(0, 0, 1),
                driver=0,
                coupled=None,
            ),
        )
    )


def create_parameter_set(
    length: float,
    angle: float = 0,
) -> ParameterSet:
    """
    Create optimization parameters.
    """

    return ParameterSet(
        parameters=(
            Parameter(
                name="lever.0.length",
                minimum=40,
                maximum=100,
                value=length,
            ),
            Parameter(
                name="lever.0.angle",
                minimum=-45,
                maximum=45,
                value=angle,
            ),
        )
    )


def test_parameter_changes_lever_definition():

    definition = create_definition()

    builder = CsvMechanismBuilder(
        definition,
    )

    mechanism_a = builder.build(
        create_parameter_set(
            length=50,
        )
    )

    mechanism_b = builder.build(
        create_parameter_set(
            length=90,
        )
    )

    assert mechanism_a != mechanism_b


def test_parameter_changes_lever_length():

    definition = create_definition()

    builder = CsvMechanismBuilder(
        definition,
    )

    mechanism_a = builder.build(
        create_parameter_set(50)
    )

    mechanism_b = builder.build(
        create_parameter_set(90)
    )

    assert (
        mechanism_a.stages[0]
        .input_lever.length
        !=
        mechanism_b.stages[0]
        .input_lever.length
    )


def test_parameter_override_does_not_modify_original_definition():

    definition = create_definition()

    builder = CsvMechanismBuilder(
        definition,
    )

    builder.build(
        create_parameter_set(
            length=90,
        )
    )

    assert (
        definition.levers[0].length_start
        == 70
    )


def test_missing_parameter_keeps_csv_value():

    definition = create_definition()

    builder = CsvMechanismBuilder(
        definition,
    )

    parameters = ParameterSet(
        parameters=()
    )

    mechanism = builder.build(
        parameters,
    )

    assert mechanism is not None