"""
solver/objective.py

Objective functions for kinematic solvers.
"""

from __future__ import annotations

from collections.abc import Callable

from mechanics.stage import Stage
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

    Returns
    -------
    Callable[[float], float]
        Function evaluating the rod length residual
        for a given output angle.
    """

    input_point = stage.input_position(
        input_angle
    )

    def residual(
        output_angle: float,
    ) -> float:

        output_point = stage.output_position(
            output_angle
        )

        return rod_length_error(
            input_point,
            output_point,
            stage.rod_length,
        )

    return residual