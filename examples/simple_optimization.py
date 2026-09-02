"""
examples/simple_optimization.py

End-to-end evolutionary optimization test.

Optimizes a single-stage mechanism against target_curve2.csv.

Issue #11: The original code referenced ``simulator`` in
the ``MechanismOptimizer`` constructor, but the variable
was never defined — only ``motion`` (a ``MotionRange``)
was created.  This would crash with a ``NameError`` at
runtime.  The fix constructs a ``MechanismSimulator``
from the ``motion`` before passing it to the optimizer.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
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
from optimization.population import (
    Population,
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


# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

TARGET_FILE = (
    BASE_DIR
    /
    "target_curve2.csv"
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
    start_angle=math.radians(-50.0),
    max_angle=math.radians(100.0),
    step=math.radians(0.1),
    direction=1,
)

# Issue #11: Construct the simulator from the motion
# range.  The original code passed an undefined
# ``simulator`` variable to MechanismOptimizer.
simulator = MechanismSimulator(
    motion=motion,
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
    builder=StandardMechanismBuilder(),
    simulator=simulator,
    fitness=fitness,
)


# -------------------------------------------------
# Parameter generation
# -------------------------------------------------

rng = random.Random(42)

def random_parameter_set() -> ParameterSet:

    return ParameterSet(
        parameters=(
            Parameter(
                name="input_lever_length",
                minimum=65.0,
                maximum=75.0,
                value=rng.uniform(
                    65.0,
                    75.0,
                ),
            ),

            Parameter(
                name="output_lever_length",
                minimum=65.0,
                maximum=75.0,
                value=rng.uniform(
                    65.0,
                    75.0,
                ),
            ),

            Parameter(
                name="input_angle_offset",
                minimum=-math.radians(45),
                maximum=math.radians(45),
                value=rng.uniform(
                    -math.radians(45),
                    math.radians(45),
                ),
            ),

            Parameter(
                name="output_angle_offset",
                minimum=-math.radians(45),
                maximum=math.radians(45),
                value=rng.uniform(
                    -math.radians(45),
                    math.radians(45),
                ),
            ),
        )
    )


# -------------------------------------------------
# Initial population
# -------------------------------------------------

population = Population(
    members=tuple(
        random_parameter_set()
        for _ in range(50)
    )
)

valid = 0

for candidate in population:

    score = optimizer.evaluate(candidate)

    if score < float("inf"):
        valid += 1

print(
    f"Valid candidates: {valid}/{len(population)}"
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
best_candidate: ParameterSet | None = None

for generation in range(4):

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
print("Best solution")
print("=============")

print(
    f"Fitness: {best_score:.12f}"
)


if best_candidate is not None:

    for name, value in (
        best_candidate.values().items()
    ):
        if "angle_offset" in name:

            print(
                f"{name}: "
                f"{math.degrees(value):.3f} deg"
            )

        else:

            print(
                f"{name}: {value:.3f}"
            )