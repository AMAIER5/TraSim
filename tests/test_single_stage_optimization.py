"""
tests/test_single_stage_optimization.py

End-to-end optimization test for a simple
two-lever mechanism.

Pipeline:

CSV
 -> Builder
 -> Simulator
 -> Fitness
 -> Evolution
"""

from __future__ import annotations

import math
import random

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve

from mechanism_io.csv_reader import CsvReader

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)

from optimization.evolution_engine import (
    EvolutionEngine,
)

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_mutation import (
    ParameterMutation,
)

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.population_factory import (
    PopulationFactory,
)

from optimization.reproduction import (
    Reproduction,
)

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)

from simulation.mechanism_simulator import (
    MechanismSimulator,
)

from simulation.motion_range import (
    MotionRange,
)


def create_parameter_template() -> ParameterSet:

    return ParameterSet(
        (
            Parameter(
                name="lever.1.length",
                minimum=20,
                maximum=40,
                value=30,
            ),
            Parameter(
                name="lever.2.length",
                minimum=80,
                maximum=120,
                value=100,
            ),
        )
    )


def test_simple_stage_optimization(
    simple_stage_csv,
    simple_target_csv,
):
    """
    Verify that evolution improves the mechanism fitness.
    """

    definition = CsvReader.read_mechanism(
        simple_stage_csv,
    )


    optimizer = MechanismOptimizer(
        builder=CsvMechanismBuilder(
            definition,
        ),
        simulator=MechanismSimulator(
            motion=MotionRange(
                start_angle=math.radians(-40),
                max_angle=math.radians(40),
                step=math.radians(5),
            ),
        ),
        fitness=CurveFitness(
            target_curve=TargetCurve.from_csv(
                simple_target_csv,
            ),
        ),
    )


    rng = random.Random(42)


    population = PopulationFactory(
        random_generator=rng,
    ).create(
        create_parameter_template(),
        size=20,
    )


    initial_score = min(
        optimizer.evaluate(candidate)
        for candidate in population
    )


    engine = EvolutionEngine(
        population=population,
        evaluator=optimizer.evaluate,
        selection_count=5,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                strength=0.1,
                random_generator=rng,
            ),
        ),
    )


    for _ in range(20):

        engine.step(
            children_count=20,
        )


    final_score = min(
        optimizer.evaluate(candidate)
        for candidate in engine.population
    )


    assert final_score < initial_score