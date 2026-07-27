"""
mechanics/stage.py

Mechanical stage consisting of two levers connected by an ideal rod.

Version 0.1:
    Geometry definition only.

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

    Parameters
    ----------
    input_lever:
        Driving lever.

    output_lever:
        Driven lever.

    rod_length:
        Fixed coupling rod length.
    """

    input_lever: Lever
    output_lever: Lever
    rod_length: float

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
        input_angle: float,
        output_angle: float,
    ) -> Stage:
        """
        Create stage from a valid reference position.

        The rod length is calculated automatically.

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

        return cls(
            input_lever=input_lever,
            output_lever=output_lever,
            rod_length=rod_length,
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
        """

        return self.input_lever.end_position(
            angle
        )

    def output_position(
        self,
        angle: float,
    ) -> Point3D:
        """
        Calculate output lever endpoint.
        """

        return self.output_lever.end_position(
            angle
        )

