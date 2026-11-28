"""
solver/objective.py

Objective functions for kinematic solvers.
"""

from __future__ import annotations

from collections.abc import Callable

from mechanics.stage import Stage
from solver._rod_residual import make_stage_residual
from solver.constraints import rod_length_error


def stage_error(
    stage: Stage,
    input_angle: float,
    output_angle: float,
) -> float:
    """
    Evaluate the rod length residual of one stage.

    The solver seeks an output angle for which

        stage_error(...) == 0

    Parameters
    ----------
    stage:
        Mechanical stage.

    input_angle:
        Input lever angle [rad].

    output_angle:
        Output lever angle [rad].
    """

    input_point = stage.input_position(
        input_angle
    )

    output_point = stage.output_position(
        output_angle
    )

    return rod_length_error(
        input_point,
        output_point,
        stage.rod_length,
    )


def create_stage_objective(
    stage: Stage,
    input_angle: float,
    *,
    output_constants: tuple[float, ...] | None = None,
) -> Callable[[float], float]:
    """
    Create a residual function for one fixed input angle.

    The input position is constant during root solving.
    It is therefore calculated once and reused for all
    output angle evaluations.

    Parameters
    ----------
    stage:
        Mechanical stage.

    input_angle:
        Fixed input lever angle [rad].

    output_constants:
        Optional precomputed scalar frame of the output lever
        (``Lever.endpoint_constants``).  Solvers that solve many
        input angles for the same stage can compute it once and
        pass it here.

    Returns
    -------
    Callable[[float], float]
        Function evaluating the rod length residual
        for a given output angle.
    """

    input_point = stage.input_position(
        input_angle
    )

    if output_constants is None:

        output_constants = (
            stage.output_lever.endpoint_constants()
        )

    return make_stage_residual(
        output_constants,
        input_point.x,
        input_point.y,
        input_point.z,
        stage.rod_length,
    )