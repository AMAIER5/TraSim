"""
examples/optimize_csv_mechanism.py

End-to-end evolutionary optimization.

Optimizes a mechanism loaded from mechanism.csv
against target_curve.csv.

User I/O angles are in DEGREES.
Internal calculations use RADIANS.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve

from mechanism_io.csv_reader import CsvReader

from mechanics.csv_mechanism_builder import CsvMechanismBuilder

from optimization.csv_parameter_factory import CsvParameterFactory
from optimization.evolution_engine import EvolutionEngine
from optimization.mechanism_optimizer import MechanismOptimizer
from optimization.parameter_mutation import ParameterMutation
from optimization.parameter_set import ParameterSet
from optimization.population_factory import PopulationFactory
from optimization.reproduction import Reproduction

from simulation.mechanism_simulator import MechanismSimulator
from simulation.motion_range import MotionRange
from simulation.stage_simulator import StageSimulator

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

# MECHANISM_FILE = BASE_DIR / "mechanism.csv"
# TARGET_FILE = BASE_DIR / "target_curve.csv"

MECHANISM_FILE = BASE_DIR / "PPBLS_mechanism.csv"
TARGET_FILE = BASE_DIR / "PBLS_target_curve_csv_mechanism.csv"

# -------------------------------------------------
# Mechanism definition
# -------------------------------------------------

definition = CsvReader.read_mechanism(MECHANISM_FILE)
builder = CsvMechanismBuilder(definition)

# -------------------------------------------------
# Optimization parameters
# -------------------------------------------------

parameter_template = CsvParameterFactory.create(definition)

# -------------------------------------------------
# Target curve
# -------------------------------------------------

target_curve = TargetCurve.from_csv(TARGET_FILE)
print(f"Target curve loaded from {TARGET_FILE}")

# Extract input angles from target curve (stored in radians in closure)
# The TargetCurve.from_csv converts CSV degrees to radians internally
target_input_angles_rad = target_curve.function.__closure__[0].cell_contents
target_output_angles_rad = target_curve.function.__closure__[1].cell_contents

# Convert to degrees for display
target_input_angles_deg = tuple(math.degrees(a) for a in target_input_angles_rad)
target_output_angles_deg = tuple(math.degrees(a) for a in target_output_angles_rad)

print(f"\nTarget curve: {len(target_input_angles_deg)} points")
for inp, out in zip(target_input_angles_deg, target_output_angles_deg):
    print(f"  {inp:.1f}° → {out:.1f}°")

# -------------------------------------------------
# Simulation setup - matches target curve input range
# -------------------------------------------------

# Use the target curve's input range for motion
# All values in radians for internal calculations
min_input_rad = min(target_input_angles_rad)
max_input_rad = max(target_input_angles_rad)
travel_range_rad = max_input_rad - min_input_rad

# For step, use 2.0 degrees converted to radians
step_rad = math.radians(2.0)

motion = MotionRange(
    start_angle=min_input_rad,
    max_angle=travel_range_rad,
    step=step_rad,
    direction=1,
)

print(f"\nMotion range: {math.degrees(min_input_rad):.1f}° to {math.degrees(max_input_rad):.1f}°, "
      f"step={math.degrees(step_rad):.1f}°")

stage_simulator = StageSimulator()
simulator = MechanismSimulator(
    motion=motion,
    stage_simulator=stage_simulator,
    stage_limit=None,
)

# -------------------------------------------------
# Fitness
# -------------------------------------------------

fitness = CurveFitness(target_curve=target_curve)

# -------------------------------------------------
# Mechanism optimizer
# -------------------------------------------------

optimizer = MechanismOptimizer(
    builder=builder,
    simulator=simulator,
    fitness=fitness,
)

# -------------------------------------------------
# Initial population
# -------------------------------------------------

rng = random.Random()
population_factory = PopulationFactory(random_generator=rng)
population = population_factory.create(parameter_template, size=100)  # Larger population

# -------------------------------------------------
# Evolution engine
# -------------------------------------------------

engine = EvolutionEngine(
    population=population,
    evaluator=optimizer.evaluate,
    selection_count=30,                # More survivors
    reproduction=Reproduction(
        mutation=ParameterMutation(
            strength=0.15,               # Stronger mutation
            random_generator=rng,
        ),
    ),
    target_fitness=0.01,              # Lower target
    max_generations=200,             # More generations
    stagnation_limit=50,             # More patience
    stagnation_tolerance=1e-5,        # Tighter tolerance
)

# -------------------------------------------------
# Initial validation
# -------------------------------------------------

engine.evaluate_population()
valid = sum(score < float("inf") for score in engine.scores.values())
print(f"Valid candidates: {valid}/{len(engine.population)}")

if valid == 0:
    raise RuntimeError("No valid candidates - mechanism is infeasible")

# -------------------------------------------------
# Evolution loop
# -------------------------------------------------

for generation in engine.run(children_count=100):
    print(f"Generation {generation:3d}: best_fitness={engine.best_score:.8f}")

# -------------------------------------------------
# Result
# -------------------------------------------------

print(f"\nEvolution finished. Reason: {engine.stop_reason}")
print(f"Best fitness: {engine.best_score:.12f}")

if engine.best_candidate is not None:
    mechanism = builder.build(engine.best_candidate)

    print("\nBest mechanism:")
    for index, stage in enumerate(mechanism.stages, start=1):
        print(f"\n  Stage {index}:")
        print(f"    Input lever:  length={stage.input_lever.length:.2f} mm, "
              f"pivot=({stage.input_lever.pivot.x:.1f}, {stage.input_lever.pivot.y:.1f}, {stage.input_lever.pivot.z:.1f})")
        print(f"    Output lever: length={stage.output_lever.length:.2f} mm, "
              f"pivot=({stage.output_lever.pivot.x:.1f}, {stage.output_lever.pivot.y:.1f}, {stage.output_lever.pivot.z:.1f})")
        print(f"    Rod length:  {stage.rod_length:.2f} mm")
        print(f"    Input range:  [{math.degrees(stage.input_angle_min):.1f}°, {math.degrees(stage.input_angle_max):.1f}°]")
        print(f"    Output range: [{math.degrees(stage.output_angle_min):.1f}°, {math.degrees(stage.output_angle_max):.1f}°]")
