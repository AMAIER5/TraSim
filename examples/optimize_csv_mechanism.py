"""
examples/optimize_csv_mechanism.py

End-to-end evolutionary optimization test.

Optimizes a mechanism loaded from mechanism.csv
against target_curve_csv_mechanism.csv.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve

from mechanism_io.csv_reader import CsvReader

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)

from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)

from optimization.evolution_engine import (
    EvolutionEngine,
)

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)

from optimization.parameter_mutation import (
    ParameterMutation,
)

from optimization.population_factory import (
    PopulationFactory,
)

from optimization.reproduction import (
    Reproduction,
)

from simulation.mechanism_simulator import (
    MechanismSimulator,
)

from simulation.motion_range import (
    MotionRange,
)

from simulation.stage_simulator import (
    StageSimulator,
)


# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

MECHANISM_FILE = (
    BASE_DIR
    /
    "mechanism.csv"
)

TARGET_FILE = (
    BASE_DIR
    /
    "target_curve_csv_mechanism.csv"
)


# -------------------------------------------------
# Mechanism definition
# -------------------------------------------------

definition = CsvReader.read_mechanism(
    MECHANISM_FILE
)

builder = CsvMechanismBuilder(
    definition
)


# -------------------------------------------------
# Optimization parameters
# -------------------------------------------------

parameter_template = (
    CsvParameterFactory.create(
        definition
    )
)


# -------------------------------------------------
# Target curve
# -------------------------------------------------

target_curve = TargetCurve.from_csv(
    TARGET_FILE
)


# -------------------------------------------------
# Simulation setup
# -------------------------------------------------

motion = MotionRange(
    start_angle=math.radians(0.0),
    max_angle=math.radians(100.0),
    step=math.radians(0.1),
    direction=-1,
)

simulator = MechanismSimulator(
    motion=motion,
    stage_simulator=StageSimulator(),
)


# -------------------------------------------------
# Fitness
# -------------------------------------------------

fitness = CurveFitness(
    target_curve=target_curve,
)


# -------------------------------------------------
# Mechanism optimizer adapter
# -------------------------------------------------

optimizer = MechanismOptimizer(
    builder=builder,
    simulator=simulator,
    fitness=fitness,
)


# -------------------------------------------------
# Initial population
# -------------------------------------------------

rng = random.Random(42)

population_factory = PopulationFactory(
    random_generator=rng,
)

population = population_factory.create(
    parameter_template,
    size=50,
)


valid = 0

for candidate in population:

    score = optimizer.evaluate(
        candidate
    )

    if score < float("inf"):

        valid += 1


print(
    f"Valid candidates: "
    f"{valid}/{len(population)}"
)


# -------------------------------------------------
# Evolution engine
# -------------------------------------------------

engine = EvolutionEngine(
    population=population,
    evaluator=optimizer.evaluate,
    selection_count=20,
    reproduction=Reproduction(
        mutation=ParameterMutation(
            strength=0.1,
            random_generator=rng,
        ),
    ),
)


# -------------------------------------------------
# Evolution loop
# -------------------------------------------------

best_score = float("inf")
best_candidate = None


for generation in range(20):

    scores = {
        candidate:
            optimizer.evaluate(candidate)
        for candidate
        in engine.population
    }

    candidate, score = min(
        scores.items(),
        key=lambda item: item[1],
    )

    if score < best_score:

        best_score = score
        best_candidate = candidate

    print(
        f"Generation {generation:3d}: "
        f"{score:.8f}"
    )

    engine.step(
        children_count=50,
    )


# -------------------------------------------------
# Result
# -------------------------------------------------

print()

print(
    "Best solution"
)

print(
    "============="
)

print(
    f"Fitness: "
    f"{best_score:.12f}"
)


if best_candidate is not None:

    for name, value in (
        best_candidate.values().items()
    ):

        print(
            f"{name}: {value:.6f}"
        )