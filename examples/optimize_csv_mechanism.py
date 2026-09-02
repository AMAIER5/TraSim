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
    stage_limit=None,
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
    target_fitness=0.01394,
    max_generations=50,
    stagnation_limit=15,
    stagnation_tolerance=1e-6,
)


# -------------------------------------------------
# Initial validation
# -------------------------------------------------

initial_scores = {
    candidate:
        optimizer.evaluate(candidate)
    for candidate
    in engine.population
}

engine.evaluate_population()

valid = sum(
    score < float("inf")
    for score in engine.scores.values()
)

print(
    f"Valid candidates: "
    f"{valid}/{len(engine.population)}"
)


# -------------------------------------------------
# Evolution loop
# -------------------------------------------------

for generation in engine.run(
    children_count=50,
):

    print(
        f"Generation {generation:3d}: "
        f"{engine.best_score:.8f}"
    )

    # print(optimizer.get_cache_stats())


if engine.best_candidate is not None:

    best_candidate = engine.best_candidate

    best_score = engine.best_score

else:

    best_candidate = None

    best_score = float("inf")
    
# -------------------------------------------------
# Result
# -------------------------------------------------
print()

print(
    "Evolution finished."
)

print(
    f"Reason: {engine.stop_reason}"
)

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

        a = stage.input_angle
        while a < 0:
            a += math.radians(360.0)
        while a > math.radians(360.0):
            a -= math.radians(360.0)
        print(
            f"  Input angle: "
            f"{math.degrees(a):.6f} deg"
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

        a = stage.output_angle
        while a < 0:
            a += math.radians(360.0)
        while a > math.radians(360.0):
            a -= math.radians(360.0)
        print(
            f"  Output angle: "
            f"{math.degrees(a):.6f} deg"
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