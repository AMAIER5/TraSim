"""
mechanics/stage.py

Mechanical stage consisting of two levers connected by an ideal rod.

Version 0.2:
    Added angular installation offsets for input and output levers.

The kinematic solution is intentionally separated.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.point3d import Point3D
from mechanics.lever import Lever


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

    input_angle_offset:
        Installation offset of input lever [rad].

    output_angle_offset:
        Installation offset of output lever [rad].
    """

    input_lever: Lever
    output_lever: Lever

    rod_length: float

    input_angle_offset: float
    output_angle_offset: float

    # Allowed motion range

    input_angle_min: float
    input_angle_max: float

    output_angle_min: float
    output_angle_max: float

    # Stored reference configuration

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
        input_angle_offset: float = 0.0,
        output_angle_offset: float = 0.0,
        input_angle_min: float = float("-inf"),
        input_angle_max: float = float("inf"),
        output_angle_min: float = float("-inf"),
        output_angle_max: float = float("inf"),
    ) -> Stage:
        """
        Create stage from a valid reference position.

        The rod length is calculated automatically.

        Angles describe shaft positions.
        Offsets describe lever installation angles.

        Formula
        -------
        rod_length =
            norm(output_endpoint - input_endpoint)
        """

        input_endpoint = (
            input_lever.end_position(
                input_angle + input_angle_offset
            )
        )

        output_endpoint = (
            output_lever.end_position(
                output_angle + output_angle_offset
            )
        )

        rod_length = (
            output_endpoint - input_endpoint
        ).norm()

        return cls(
            input_lever=input_lever,
            output_lever=output_lever,
            rod_length=rod_length,

            input_angle_offset=input_angle_offset,
            output_angle_offset=output_angle_offset,

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
        Calculate input lever endpoint.

        The installation offset is applied automatically.
        """

        return self.input_lever.end_position(
            angle + self.input_angle_offset
        )

    def output_position(
        self,
        angle: float,
    ) -> Point3D:
        """
        Calculate output lever endpoint.

        The installation offset is applied automatically.
        """

        return self.output_lever.end_position(
            angle + self.output_angle_offset
        )
        
    def accepts_input_angle(
        self,
        angle: float,
    ) -> bool:
        """
        Check whether an input angle lies inside
        the defined mechanical working range.
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
        Check whether an output angle lies inside
        the defined mechanical working range.
        """

        return (
            self.output_angle_min
            <= angle
            <= self.output_angle_max
        )