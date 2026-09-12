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
from optimization.mechanism_builder import MechanismBuilder

class StandardMechanismBuilder(MechanismBuilder):
    """
    Builds a standard single-stage mechanism.

    Expected parameters:

    input_lever_length
    output_lever_length
    input_angle
    output_angle

    The lever installation poses (reference directions) are
    fixed by this builder: both levers start in the +X
    direction, so the lever angles supplied by the optimizer
    are measured relative to that installation pose.  The
    angle limits are left unbounded, so the build-space
    constraint is governed by the motion range and the
    solver's kinematic feasibility rather than by explicit
    lever-angle segments here.

    All angles are stored internally in radians.
    """

    def build(self, parameters: ParameterSet) -> Mechanism:
        """
        Build a mechanism from optimization parameters.

        The rod length is calculated automatically
        from the reference position.

        Reference position:
            input_angle  = the supplied input lever angle
            output_angle = the supplied output lever angle

        The lever installation pose is the +X direction, so
        the supplied angles are lever angles relative to that
        pose.
        """

        input_length = parameters.get(
            "input_lever_length"
        ).value

        output_length = parameters.get(
            "output_lever_length"
        ).value

        input_angle = parameters.get(
            "input_angle"
        ).value

        output_angle = parameters.get(
            "output_angle"
        ).value

        rotation_axis = Vector3D(
            0.0,
            0.0,
            1.0,
        )

        reference_direction = Vector3D(
            1.0,
            0.0,
            0.0,
        )

        input_lever = Lever(
            pivot=Point3D(
                0.0,
                0.0,
                0.0,
            ),
            axis=rotation_axis,
            length=input_length,
            reference_direction=reference_direction,
        )

        output_lever = Lever(
            pivot=Point3D(
                100.0,
                0.0,
                0.0,
            ),
            axis=rotation_axis,
            length=output_length,
            reference_direction=reference_direction,
        )

        stage = Stage.from_reference_position(
            input_lever=input_lever,
            output_lever=output_lever,
            input_angle=input_angle,
            output_angle=output_angle,
        )

        return Mechanism(
            stages=(
                stage,
            )
        )

    def get_validation_results(
        self,
    ) -> tuple:
        """
        Return the stage validation results from the most recent build.

        The standard builder performs no validation, so an empty
        tuple is returned.
        """

        return ()
