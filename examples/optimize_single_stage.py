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

from analysis.target_curve import TargetCurve

from optimization.curve_fitness import (
    CurveFitness,
)

from optimization.optimization_pipeline import (
    OptimizationPipeline,
)

from optimization.optimization_problem import (
    OptimizationProblem,
)

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_set import (
    ParameterSet,
)

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)

from optimization.mechanism_simulator import (
    MechanismSimulator,
)

from simulation.stage_simulator import (
    StageSimulator,
)

# Import Stage enum for correct typing when constructing StageSimulator
from mechanics.stage import (
    Stage,
)


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


def create_target_curve():

    #
    # Identity transfer
    #
    # output = input
    #

    return TargetCurve(
        function=lambda angle: angle,
    )


def main():

    # use the Stage enum instead of a raw int to satisfy type requirements
    stage_simulator = StageSimulator()

    mechanism_simulator = (
        MechanismSimulator(
            stage_simulator=stage_simulator.run,
        )
    )

    fitness = CurveFitness(
        target_curve=create_target_curve(),
    )

    problem = OptimizationProblem(

        parameter_template=(
            create_parameter_template()
        ),

        builder=StandardMechanismBuilder(),

        simulator=lambda mechanism:
            mechanism_simulator.simulate(
                mechanism
            )[0],

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