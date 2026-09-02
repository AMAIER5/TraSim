"""
tests/test_multistage_curve_fitness.py
"""

from __future__ import annotations

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve
from simulation.simulation_result import SimulationResult


def test_curve_fitness_uses_last_stage_output():
    """
    Fitness must evaluate the transfer curve of the final stage,
    not the first stage.
    """

    simulation = (
        SimulationResult(
            input_angles=(
                0.0,
                1.0,
                2.0,
            ),
            output_angles=(
                10.0,
                11.0,
                12.0,
            ),
            success=True,
        ),
        SimulationResult(
            input_angles=(
                10.0,
                11.0,
                12.0,
            ),
            output_angles=(
                20.0,
                21.0,
                22.0,
            ),
            success=True,
        ),
    )

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle + 10.0,
        ),
    )

    result = fitness.evaluate(
        simulation,
    )

    assert result == 0.0


def test_curve_fitness_does_not_use_first_stage_output():
    """
    A perfect second stage must remain perfect even if
    the first stage has an unrelated transfer curve.
    """

    simulation = (
        SimulationResult(
            input_angles=(
                0.0,
                1.0,
                2.0,
            ),
            output_angles=(
                100.0,
                100.0,
                100.0,
            ),
            success=True,
        ),
        SimulationResult(
            input_angles=(
                0.0,
                1.0,
                2.0,
            ),
            output_angles=(
                5.0,
                6.0,
                7.0,
            ),
            success=True,
        ),
    )

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle + 5.0,
        ),
    )

    result = fitness.evaluate(
        simulation,
    )

    assert result == 0.0