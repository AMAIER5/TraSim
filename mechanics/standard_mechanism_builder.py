"""
mechanics/standard_mechanism_builder.py

Standard mechanism construction from
optimization parameters.
"""

from __future__ import annotations

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.lever import Lever
from mechanics.rod import Rod
from mechanics.stage import Stage
from mechanics.mechanism import Mechanism

from optimization.parameter_set import ParameterSet


class StandardMechanismBuilder:
    """
    Builds a standard single-stage mechanism.

    Expected parameters:

    input_lever_length
    output_lever_length
    rod_length
    """

    def build(
        self,
        parameters: ParameterSet,
    ) -> Mechanism:
        """
        Create mechanism from parameters.
        """

        input_length = parameters.get(
            "input_lever_length"
        ).value

        output_length = parameters.get(
            "output_lever_length"
        ).value

        rod_length = parameters.get(
            "rod_length"
        ).value

        rotation_axis = Vector3D(
            0.0,
            0.0,
            1.0,
        )

        input_lever = Lever(
            pivot=Point3D(
                0.0,
                0.0,
                0.0,
            ),
            axis=rotation_axis,
            length=input_length,
        )

        output_lever = Lever(
            pivot=Point3D(
                100.0,
            0.0,
                0.0,
            ),
            axis=rotation_axis,
            length=output_length,
        )

        input_endpoint = input_lever.end_position(
            0.0
        )

        output_endpoint = output_lever.end_position(
            0.0
        )

        rod = Rod(
            point_a=input_endpoint,
            point_b=output_endpoint,
            length=rod_length,
        )

        stage = Stage(
            input_lever=input_lever,
            output_lever=output_lever,
            rod_length=rod_length,
            input_angle=0.0,
            output_angle=0.0,
            input_endpoint=input_endpoint,
            output_endpoint=output_endpoint,
        )

        return Mechanism(
            stages=(
                stage,
            )
        )