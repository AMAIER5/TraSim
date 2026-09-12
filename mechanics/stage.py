"""
mechanics/stage.py

Mechanical stage consisting of two levers connected by an ideal rod.

Version 0.3:
    Removed angular installation offsets.  The angles flowing
    through a stage are now lever angles: each angle is measured
    relative to the corresponding lever's ``reference_direction``.
    The input/output angle limits therefore directly describe the
    admissible lever-angle segment (circular build space) of each
    lever, instead of constraining the shaft rotation while the
    lever sat at an unspecified offset.

The kinematic solution is intentionally separated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.point3d import Point3D
from mechanics.lever import Lever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Stage:
    """
    One mechanical linkage stage.

    A stage consists of two levers connected by
    an ideal coupling rod.

    Parameters
    ----------
    input_lever:
        Driving lever.

    output_lever:
        Driven lever.

    rod_length:
        Fixed coupling rod length.

    input_angle_min:
        Minimum admissible input lever angle [rad], measured
        relative to the input lever's ``reference_direction``.

    input_angle_max:
        Maximum admissible input lever angle [rad].

    output_angle_min:
        Minimum admissible output lever angle [rad], measured
        relative to the output lever's ``reference_direction``.

    output_angle_max:
        Maximum admissible output lever angle [rad].
    """

    input_lever: Lever
    output_lever: Lever

    rod_length: float

    input_angle_min: float
    input_angle_max: float

    output_angle_min: float
    output_angle_max: float

    input_angle: float
    output_angle: float
    input_endpoint: Point3D
    output_endpoint: Point3D

    @classmethod
    def from_reference_position(
        cls,
        input_lever: Lever,
        output_lever: Lever,
        input_angle: float = 0.0,
        output_angle: float = 0.0,
        input_angle_min: float = float("-inf"),
        input_angle_max: float = float("inf"),
        output_angle_min: float = float("-inf"),
        output_angle_max: float = float("inf"),
        validate_reference: bool = True,
    ) -> Stage:
        """
        Create stage from a valid reference position.

        The rod length is calculated automatically.

        Angles are lever angles, each measured relative to the
        corresponding lever's ``reference_direction``.  The angle
        limits describe the admissible lever-angle segment of each
        lever (its circular build space).

        Issue #7: When ``validate_reference`` is True (the
        default), the reference angles are checked against
        the supplied angle ranges and a ``ValueError`` is
        raised if they lie outside.  Set
        ``validate_reference=False`` to create a stage with
        an intentionally impossible configuration (used by
        tests of the motion validator).

        Formula
        -------
        rod_length =
            norm(output_endpoint - input_endpoint)
        """

        input_endpoint = (
            input_lever.end_position(
                input_angle
            )
        )

        output_endpoint = (
            output_lever.end_position(
                output_angle
            )
        )

        rod_length = (
            output_endpoint - input_endpoint
        ).norm()

        if validate_reference:

            if not (
                input_angle_min
                <= input_angle
                <= input_angle_max
            ):
                raise ValueError(
                    f"Reference input angle {input_angle} "
                    f"is outside the allowed range "
                    f"[{input_angle_min}, "
                    f"{input_angle_max}]."
                )

            if not (
                output_angle_min
                <= output_angle
                <= output_angle_max
            ):
                raise ValueError(
                    f"Reference output angle {output_angle} "
                    f"is outside the allowed range "
                    f"[{output_angle_min}, "
                    f"{output_angle_max}]."
                )

        return cls(
            input_lever=input_lever,
            output_lever=output_lever,
            rod_length=rod_length,

            input_angle_min=input_angle_min,
            input_angle_max=input_angle_max,

            output_angle_min=output_angle_min,
            output_angle_max=output_angle_max,

            input_angle=input_angle,
            output_angle=output_angle,

            input_endpoint=input_endpoint,
            output_endpoint=output_endpoint,
        )

    def input_position(
        self,
        angle: float,
    ) -> Point3D:
        """
        Calculate input lever endpoint for a lever angle.
        """

        return self.input_lever.end_position(
            angle
        )

    def output_position(
        self,
        angle: float,
    ) -> Point3D:
        """
        Calculate output lever endpoint for a lever angle.
        """

        return self.output_lever.end_position(
            angle
        )

    def accepts_input_angle(
        self,
        angle: float,
    ) -> bool:
        """
        Check whether an input lever angle lies inside
        the admissible lever-angle segment.
        """

        return (
            self.input_angle_min
            <= angle
            <= self.input_angle_max
        )

    def accepts_output_angle(
        self,
        angle: float,
    ) -> bool:
        """
        Check whether an output lever angle lies inside
        the admissible lever-angle segment.
        """

        return (
            self.output_angle_min
            <= angle
            <= self.output_angle_max
        )
