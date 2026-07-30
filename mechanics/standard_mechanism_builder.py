"""
mechanics/standard_mechanism_builder.py

Standard mechanism construction from
optimization parameters.
"""

from __future__ import annotations

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from optimization.parameter_set import ParameterSet


class StandardMechanismBuilder:
    """
    Builds a standard single-stage mechanism.

    Expected parameters:

    input_lever_length
    output_lever_length
    input_angle_offset
    output_angle_offset

    All angles are stored internally in radians.
    """

    def build(
        self,
        parameters: ParameterSet,
    ) -> Mechanism:
        """
        Build a mechanism from optimization parameters.

        The rod length is calculated automatically
        from the reference position.

        Reference position:
            input_angle  = 0 rad
            output_angle = 0 rad

        The lever installation offsets are included
        in the reference geometry.
        """

        input_length = parameters.get(
            "input_lever_length"
        ).value

        output_length = parameters.get(
            "output_lever_length"
        ).value

        input_angle_offset = parameters.get(
            "input_angle_offset"
        ).value

        output_angle_offset = parameters.get(
            "output_angle_offset"
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

        stage = Stage.from_reference_position(
            input_lever=input_lever,
            output_lever=output_lever,
            input_angle=0.0,
            output_angle=0.0,
            input_angle_offset=input_angle_offset,
            output_angle_offset=output_angle_offset,
        )

        return Mechanism(
            stages=(
                stage,
            )
        )