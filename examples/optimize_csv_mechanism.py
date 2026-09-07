"""
examples/optimize_csv_mechanism.py

End-to-end evolutionary optimization.

Optimizes a mechanism loaded from mechanism.csv
against target_curve.csv.

User I/O angles are in DEGREES.
Internal calculations use RADIANS.

Rod lengths are AUTO-CALCULATED from lever endpoints at reference angles.
The solver will naturally find non-blocking solutions if they exist in the search space.

For stable optimization:
- Use mutation strength of 0.01-0.05 (not 0.15)
- Use sufficient generations (500+)
- Use tight stagnation tolerance (1e-8)
"""

from __future__ import annotations

import csv
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

# For your simple 1:1 example:
MECHANISM_FILE = BASE_DIR / "mechanism.csv"
TARGET_FILE = BASE_DIR / "target_curve.csv"

# For your PPBLS example (comment out the above lines if needed):
# MECHANISM_FILE = BASE_DIR / "PPBLS_mechanism.csv"
# TARGET_FILE = BASE_DIR / "PBLS_target_curve_csv_mechanism.csv"

# -------------------------------------------------
# Mechanism definition
# -------------------------------------------------

definition = CsvReader.read_mechanism(MECHANISM_FILE)
builder = CsvMechanismBuilder(definition)

# -------------------------------------------------
# Print mechanism info
# -------------------------------------------------

print("=" * 80)
print("LOADING MECHANISM")
print("=" * 80)
print(f"\nMechanism: {len(definition.levers)} levers")
for lever in definition.levers:
    driver_str = f", driver={lever.driver}" if lever.driver else ""
    coupled_str = f", coupled={lever.coupled}" if lever.coupled else ""
    print(f"  Lever {lever.id}: pivot=({lever.pivot.x:.1f}, {lever.pivot.y:.1f}, {lever.pivot.z:.1f}), "
          f"axis=({lever.axis.x:.1f}, {lever.axis.y:.1f}, {lever.axis.z:.1f}), "
          f"length={lever.length_start:.1f}mm, "
          f"angle=[{math.degrees(lever.angle_min):.1f}\u00b0, {math.degrees(lever.angle_max):.1f}\u00b0]"
          f"{driver_str}{coupled_str}")

# -------------------------------------------------
# Target curve - READ DIRECTLY FROM CSV
# -------------------------------------------------

target_input_angles_deg = []
target_output_angles_deg = []

with open(TARGET_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        target_input_angles_deg.append(float(row['input_angle']))
        target_output_angles_deg.append(float(row['output_angle']))

target_input_angles_rad = tuple(math.radians(a) for a in target_input_angles_deg)
target_output_angles_rad = tuple(math.radians(a) for a in target_output_angles_deg)

target_curve = TargetCurve.from_csv(TARGET_FILE)

print("\n" + "=" * 80)
print("LOADING TARGET CURVE")
print("=" * 80)
print(f"\nTarget curve: {len(target_input_angles_deg)} points")
for inp, out in zip(target_input_angles_deg, target_output_angles_deg):
    print(f"  {inp:.1f}\u00b0 \u2192 {out:.1f}\u00b0")

# -------------------------------------------------
# Simulation setup
# -------------------------------------------------

min_input_rad = min(target_input_angles_rad)
max_input_rad = max(target_input_angles_rad)
travel_range_rad = max_input_rad - min_input_rad
step_rad = math.radians(2.0)

motion = MotionRange(
    start_angle=min_input_rad,
    max_angle=travel_range_rad,
    step=step_rad,
    direction=1,
)

print("\n" + "=" * 80)
print("OPTIMIZATION SETUP")
print("=" * 80)
print(f"\nParameters ({len(definition.levers) * 2}):")

parameter_template = CsvParameterFactory.create(definition)

for param in parameter_template.parameters:
    if "angle" in param.name:
        min_deg = math.degrees(param.minimum)
        max_deg = math.degrees(param.maximum)
        val_deg = math.degrees(param.value)
        print(f"  {param.name}: [{min_deg:.1f}\u00b0, {max_deg:.1f}\u00b0], default={val_deg:.1f}\u00b0")
    else:
        print(f"  {param.name}: [{param.minimum:.1f}, {param.maximum:.1f}], default={param.value:.1f}")

print(f"\nMotion range: {math.degrees(min_input_rad):.1f}\u00b0 to {math.degrees(max_input_rad):.1f}\u00b0, "
      f"step=2.0\u00b0")

# -------------------------------------------------
# Create simulator and optimizer
# -------------------------------------------------

stage_simulator = StageSimulator()
simulator = MechanismSimulator(
    motion=motion,
    stage_simulator=stage_simulator,
    stage_limit=None,
)

# FIX (4): Increased patience for optimization
fitness = CurveFitness(
    target_curve=target_curve,
    motion_start=min_input_rad,
    motion_range=travel_range_rad,
)
optimizer = MechanismOptimizer(
    builder=builder,
    simulator=simulator,
    fitness=fitness,
)

# -------------------------------------------------
# Initial population
# -------------------------------------------------

rng = random.Random(42)  # Fixed seed for reproducibility
population_factory = PopulationFactory(random_generator=rng)
population = population_factory.create(parameter_template, size=100)

# -------------------------------------------------
# Evolution engine - FIXES (2), (3), (4)
# -------------------------------------------------

# FIX (2): Geometric stability - use smaller population for more focused search
# FIX (3): Reduced mutation strength from 0.15 to 0.01 for finer, more stable search
# FIX (4): Increased patience - more generations and tighter tolerance
engine = EvolutionEngine(
    population=population,
    evaluator=optimizer.evaluate,
    selection_count=30,
    reproduction=Reproduction(
        mutation=ParameterMutation(
            strength=0.01,               # FIX (3): Reduced from 0.15 to 0.01
            random_generator=rng,
        ),
    ),
    target_fitness=0.01,
    max_generations=1000,             # FIX (4): Increased from 200 to 1000
    stagnation_limit=200,             # FIX (4): Increased from 50 to 200
    stagnation_tolerance=1e-8,        # FIX (4): Tighter from 1e-5 to 1e-8
)

# -------------------------------------------------
# Initial validation
# -------------------------------------------------

print("\n" + "=" * 80)
print("EVOLUTION ENGINE")
print("=" * 80)

engine.evaluate_population()
valid = sum(score < float("inf") for score in engine.scores.values())

print("\n" + "=" * 80)
print("INITIAL POPULATION")
print("=" * 80)
print(f"Valid candidates: {valid}/{len(engine.population)}")
if engine.best_score < float("inf"):
    print(f"Best initial fitness: {engine.best_score:.8f}")
else:
    print("Best initial fitness: inf (no valid candidates)")

if valid == 0:
    raise RuntimeError("No valid candidates - mechanism is infeasible")

# -------------------------------------------------
# Evolution loop
# -------------------------------------------------

print("\n" + "=" * 80)
print("EVOLUTION PROGRESS")
print("=" * 80)
print("  Generation |         Fitness |  Improvement")
print("-" * 55)

prev_best = float("inf")
generation_count = 0

for generation in engine.run(children_count=100):
    generation_count += 1
    improvement = prev_best - engine.best_score
    prev_best = engine.best_score
    if generation == 0:
        improvement_str = "          inf"
    else:
        improvement_str = f"{improvement:>12.8f}"
    print(f"  {generation:3d} | {engine.best_score:>16.8f} | {improvement_str}")

# -------------------------------------------------
# Results
# -------------------------------------------------

print("\n" + "=" * 80)
print("OPTIMIZATION RESULTS")
print("=" * 80)

print(f"\nStop reason: {engine.stop_reason}")
print(f"Best fitness: {engine.best_score:.12f}")
print(f"Generations run: {generation_count}")

cache_stats = optimizer.get_cache_stats()
print(f"\nCache statistics:")
print(f"  Evaluations: {cache_stats['evaluations']}")
print(f"  Cache hits: {cache_stats['cache_hits']}")
print(f"  Cache misses: {cache_stats['cache_misses']}")
print(f"  Cache size: {cache_stats['cache_size']}")

# -------------------------------------------------
# Best mechanism details
# -------------------------------------------------

if engine.best_candidate is not None:
    mechanism = builder.build(engine.best_candidate)

    print("\n" + "=" * 80)
    print("BEST MECHANISM")
    print("=" * 80)
    print("\nStages:")
    for index, stage in enumerate(mechanism.stages, start=1):
        print(f"\n  Stage {index}:")
        print(f"    Input lever:  length={stage.input_lever.length:.2f} mm, "
              f"pivot=({stage.input_lever.pivot.x:.1f}, {stage.input_lever.pivot.y:.1f}, {stage.input_lever.pivot.z:.1f})")
        print(f"    Output lever: length={stage.output_lever.length:.2f} mm, "
              f"pivot=({stage.output_lever.pivot.x:.1f}, {stage.output_lever.pivot.y:.1f}, {stage.output_lever.pivot.z:.1f})")
        print(f"    Rod length:  {stage.rod_length:.2f} mm")
        print(f"    Input range:  [{math.degrees(stage.input_angle_min):.1f}\u00b0, {math.degrees(stage.input_angle_max):.1f}\u00b0]")
        print(f"    Output range: [{math.degrees(stage.output_angle_min):.1f}\u00b0, {math.degrees(stage.output_angle_max):.1f}\u00b0]")

    # Validation results
    validation_results = builder.get_validation_results()
    print("\n" + "-" * 55)
    print("STAGE VALIDATION RESULTS")
    print("-" * 55)
    for stage_idx, result in enumerate(validation_results, start=1):
        print(f"\n  Stage {stage_idx}:")
        print(f"    Valid: {result.valid}")
        if not result.valid:
            print(f"    Reason: {result.reason}")
            if hasattr(result, 'failed_at_input_angle') and result.failed_at_input_angle is not None:
                print(f"    Failed at: {math.degrees(result.failed_at_input_angle):.1f}\u00b0")

    # Simulation results
    print("\n" + "-" * 55)
    print("SIMULATION RESULTS")
    print("-" * 55)

    simulation_results = simulator.simulate(mechanism)

    for stage_index, stage_result in enumerate(simulation_results, start=1):
        print(f"\n  Stage {stage_index}:")
        print(f"    Success: {stage_result.success}")
        print(f"    Input points: {len(stage_result.input_angles)}")
        if stage_result.success:
            print(f"    Output points: {len(stage_result.output_angles)}")
            if stage_result.output_angles:
                output_deg = tuple(math.degrees(a) for a in stage_result.output_angles)
                print(f"    Output range: [{min(output_deg):.1f}\u00b0, {max(output_deg):.1f}\u00b0]")
        else:
            if stage_result.blocked_at is not None:
                print(f"    Blocked at: {math.degrees(stage_result.blocked_at):.1f}\u00b0")

    # Solver statistics
    print("\n" + "=" * 80)
    print("SOLVER STATISTICS")
    print("=" * 80)
    total_stats = {}
    for solver in stage_simulator.solvers:
        stats = solver.get_stats()
        for key, value in stats.items():
            total_stats[key] = total_stats.get(key, 0) + value
    for key, value in sorted(total_stats.items()):
        print(f"  {key:>25}: {value:>10}")
