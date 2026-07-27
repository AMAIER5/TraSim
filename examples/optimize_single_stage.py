"""
examples/optimize_single_stage.py

First complete optimization example.

This example demonstrates the complete workflow:

Parameter template
    ->
Optimization
    ->
Best mechanism
"""

from __future__ import annotations

import math
import random

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve
from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)
from optimization.optimization_pipeline import (
    OptimizationPipeline,
)
from optimization.optimization_problem import (
    OptimizationProblem,
)
from optimization.parameter import Parameter
from optimization.parameter_set import ParameterSet
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange
from simulation.stage_simulator import StageSimulator


def create_parameter_template() -> ParameterSet:

    return ParameterSet(
        (
            Parameter(
                name="input_lever_length",
                minimum=20.0,
                maximum=80.0,
                value=40.0,
            ),
            Parameter(
                name="output_lever_length",
                minimum=20.0,
                maximum=80.0,
                value=40.0,
            ),
            Parameter(
                name="rod_length",
                minimum=60.0,
                maximum=180.0,
                value=120.0,
            ),
        )
    )


def create_target_curve() -> TargetCurve:

    return TargetCurve(
        function=lambda angle: angle,
    )


def create_motion() -> MotionRange:

    return MotionRange(
        start_angle=0.0,
        max_angle=math.radians(90.0),
        step=math.radians(1.0),
    )


def main() -> None:

    mechanism_simulator = MechanismSimulator(
        motion=create_motion(),
        stage_simulator=StageSimulator(),
    )

    fitness = CurveFitness(
        target_curve=create_target_curve(),
    )

    problem = OptimizationProblem(
        parameter_template=create_parameter_template(),
        builder=StandardMechanismBuilder(),
        simulator=mechanism_simulator,
        fitness=fitness,
        random_generator=random.Random(1),
    )

    pipeline = OptimizationPipeline(
        problem=problem,
    )

    population = pipeline.run(
        population_size=30,
        generations=50,
        children_per_generation=20,
    )

    best = population[0]

    print()
    print("Optimization finished")
    print("---------------------")
    print()

    for parameter in best.parameters:
        print(
            f"{parameter.name:25}"
            f"{parameter.value:8.3f}"
        )


if __name__ == "__main__":
    main()