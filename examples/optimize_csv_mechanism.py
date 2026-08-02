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

print(target_curve)

# -------------------------------------------------
# Simulation setup
# -------------------------------------------------

motion = MotionRange(
    start_angle=math.radians(0.0),
    max_angle=math.radians(100.0),
    step=math.radians(0.1),
    direction=-1,
)

stage_simulator = StageSimulator()

simulator = MechanismSimulator(
    motion=motion,
    stage_simulator=stage_simulator,
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


for generation in range(5):

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


# -------------------------------------------------
# Construction data
# -------------------------------------------------

if best_candidate is not None:

    mechanism = builder.build(
        best_candidate
    )

    print()

    print(
        "Construction data"
    )

    print(
        "-----------------"
    )

    for index, stage in enumerate(
        mechanism.stages,
        start=1,
    ):

        print()

        print(
            f"Stage {index}"
        )

        print(
            f"  Input lever length: "
            f"{stage.input_lever.length:.6f} mm"
        )

        print(
            f"  Input angle: "
            f"{math.degrees(stage.input_angle):.6f} deg"
        )

        print(
            f"  Input angle offset: "
            f"{math.degrees(stage.input_angle_offset):.6f} deg"
        )

        print()

        print(
            f"  Output lever length: "
            f"{stage.output_lever.length:.6f} mm"
        )

        print(
            f"  Output angle: "
            f"{math.degrees(stage.output_angle):.6f} deg"
        )

        print(
            f"  Output angle offset: "
            f"{math.degrees(stage.output_angle_offset):.6f} deg"
        )

        print()

        print(
            f"  Rod length: "
            f"{stage.rod_length:.6f} mm"
        )
        
# -------------------------------------------------
# Solver statistics
# -------------------------------------------------

print()

print(
    "Solver statistics"
)

print(
    "================="
)

total_stats: dict[str, int] = {}

for solver in stage_simulator.solvers:

    if not hasattr(solver, "get_stats"):
        continue

    for key, value in solver.get_stats().items():

        total_stats[key] = (
            total_stats.get(key, 0)
            +
            value
        )

if total_stats:

    for key, value in total_stats.items():

        print(
            f"{key}: {value}"
        )

else:

    print(
        "No statistics available."
    )