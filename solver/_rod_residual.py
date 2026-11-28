"""
solver/_rod_residual.py

Allocation-free scalar kinematics for the solver hot loop.

This module is the pure-Python reference implementation.  The same
functions are also available as a Cython-transpiled native extension:
build ``solver/_rod_residual.pyx`` in place (see docs/performance.md)
and the compiled module takes import precedence over this file
automatically, so no code changes are needed to activate it.

All functions perform exactly the same floating-point operations in
the same order as the object-based path
(``Lever.end_position`` + ``solver.constraints.rod_length_error``),
so results are bit-identical.
"""

from __future__ import annotations

from collections.abc import Callable
from math import cos, sin, sqrt


def rod_length_residual(
    output_x: float,
    output_y: float,
    output_z: float,
    input_x: float,
    input_y: float,
    input_z: float,
    rod_length: float,
) -> float:
    """
    Rod length residual from plain float endpoint components.

    Formula
    -------
    error = sqrt(dx*dx + dy*dy + dz*dz) - rod_length

    where (dx, dy, dz) = output endpoint - input endpoint.
    """

    dx = output_x - input_x

    dy = output_y - input_y

    dz = output_z - input_z

    return sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    ) - rod_length


def stage_residual(
    angle: float,
    pivot_x: float,
    pivot_y: float,
    pivot_z: float,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    reference_x: float,
    reference_y: float,
    reference_z: float,
    cross_x: float,
    cross_y: float,
    cross_z: float,
    dot: float,
    length: float,
    input_x: float,
    input_y: float,
    input_z: float,
    rod_length: float,
) -> float:
    """
    Fused lever endpoint + rod length residual in one call.

    ``angle`` is the output lever angle [rad].  The remaining
    arguments are the constant scalar frame of the output lever as
    returned by ``Lever.endpoint_constants`` followed by the fixed
    input endpoint and the rod length.

    The endpoint is evaluated with the Rodrigues formula, exactly
    as in ``Lever.end_position_components``:

        direction = reference * cos(angle)
                  + (axis x reference) * sin(angle)
                  + axis * (axis . reference) * (1 - cos(angle))
        endpoint  = pivot + direction * length
    """

    c = cos(angle)
    s = sin(angle)

    direction_x = (
        reference_x * c
        + cross_x * s
        + axis_x * dot * (1.0 - c)
    )

    direction_y = (
        reference_y * c
        + cross_y * s
        + axis_y * dot * (1.0 - c)
    )

    direction_z = (
        reference_z * c
        + cross_z * s
        + axis_z * dot * (1.0 - c)
    )

    return rod_length_residual(
        pivot_x + direction_x * length,
        pivot_y + direction_y * length,
        pivot_z + direction_z * length,
        input_x,
        input_y,
        input_z,
        rod_length,
    )


def make_stage_residual(
    constants: tuple[float, ...],
    input_x: float,
    input_y: float,
    input_z: float,
    rod_length: float,
) -> Callable[[float], float]:
    """
    Create a residual function for one fixed input position.

    ``constants`` is the output lever frame as returned by
    ``Lever.endpoint_constants``.  The returned closure evaluates
    ``stage_residual`` without repacking the constants on every
    call.
    """

    def residual(
        output_angle: float,
    ) -> float:

        return stage_residual(
            output_angle,
            *constants,
            input_x,
            input_y,
            input_z,
            rod_length,
        )

    return residual
