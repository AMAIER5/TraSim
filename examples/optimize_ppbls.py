"""
examples/optimize_ppbls.py

End-to-end evolutionary optimization for the PPBLS mechanism.

Optimizes a mechanism loaded from PPBLS_mechanism.csv
against PBLS_target_curve_csv_mechanism.csv.

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

from simulation.point_motion import PointMotion

from mechanics.csv_mechanism_builder import CsvMechanismBuilder

from optimization.csv_parameter_factory import CsvParameterFactory
from optimization.adaptive_strength import AdaptiveStrength
from optimization.crossover import Crossover
from optimization.evolution_engine import EvolutionEngine
from optimization.mechanism_optimizer import MechanismOptimizer
from optimization.parameter_mutation import ParameterMutation
from optimization.parameter_set import ParameterSet
from optimization.population_factory import PopulationFactory
from optimization.reproduction import Reproduction

from simulation.mechanism_simulator import MechanismSimulator
from simulation.stage_simulator import StageSimulator

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

# PPBLS example files:
MECHANISM_FILE = BASE_DIR / "PPBLS_mechanism.csv"
TARGET_FILE = BASE_DIR / "PBLS_target_curve_csv_mechanism.csv"

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
          f"angle=[{math.degrees(lever.angle_min):.1f}deg, {math.degrees(lever.angle_max):.1f}deg]"
          f"{driver_str}{coupled_str}")

# -------------------------------------------------
# HEBEL-UEBERSICHT (INITIAL) - complete lever overview
# for physical reconstruction, read from the mechanism
# definition (definition.levers[i].angle_start).
# -------------------------------------------------

print("\n" + "=" * 80)
print("HEBEL-UEBERSICHT (INITIAL)")
print("=" * 80)
print("  Vollstaendige Uebersicht aller Hebel aus der Mechanismusdefinition.")
print("  angle_start ist der initiale Winkel JEDES Hebels in Grad.\n")

for lever in definition.levers:
    driver_str = f"driver={lever.driver}" if lever.driver is not None else "-"
    coupled_str = f"coupled={lever.coupled}" if lever.coupled is not None else "-"
    chain = driver_str if lever.driver is not None else coupled_str
    print(f"  Lever {lever.id}:")
    print(f"    Drehpunkt (pivot) : ({lever.pivot.x:.3f}, {lever.pivot.y:.3f}, {lever.pivot.z:.3f})")
    print(f"    Achse (axis)      : ({lever.axis.x:.3f}, {lever.axis.y:.3f}, {lever.axis.z:.3f})")
    print(f"    Winkelbereich      : [{math.degrees(lever.angle_min):.3f}deg, {math.degrees(lever.angle_max):.3f}deg]")
    print(f"    INITIALER WINKEL   : {math.degrees(lever.angle_start):.3f}deg   (angle_start)")
    print(f"    Laengenbereich     : [{lever.length_min:.3f}, {lever.length_max:.3f}] mm")
    print(f"    length_start       : {lever.length_start:.3f} mm")
    print(f"    Verkettung         : {chain}")
    print()

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

# Non-interpolating target curve: fitness is only defined at the
# 11 prescribed support points; no linear interpolation between
# them is used for evaluation.
target_curve = TargetCurve.from_csv_strict(TARGET_FILE)

print("\n" + "=" * 80)
print("LOADING TARGET CURVE")
print("=" * 80)
print(f"\nTarget curve: {len(target_input_angles_deg)} points")
for inp, out in zip(target_input_angles_deg, target_output_angles_deg):
    print(f"  {inp:.1f}deg -> {out:.1f}deg")

# -------------------------------------------------
# Simulation setup
# -------------------------------------------------
#
# Simulate exactly at the prescribed support points of the target
# curve (the 11 input angles from the CSV), not on a 2-degree grid.
# This keeps the chain intact (output of a stage feeds the next
# stage, see MechanismSimulator.simulate) while ensuring the
# fitness is compared only at the support points.

min_input_rad = min(target_input_angles_rad)
max_input_rad = max(target_input_angles_rad)
travel_range_rad = max_input_rad - min_input_rad

motion = PointMotion(
    angles=target_input_angles_rad,
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
        print(f"  {param.name}: [{min_deg:.1f}deg, {max_deg:.1f}deg], default={val_deg:.1f}deg")
    else:
        print(f"  {param.name}: [{param.minimum:.1f}, {param.maximum:.1f}], default={param.value:.1f}")

print(f"\nMotion: simulated at the {len(target_input_angles_rad)} prescribed "
      f"support points ({math.degrees(min_input_rad):.1f}deg to "
      f"{math.degrees(max_input_rad):.1f}deg), no interpolation")

# -------------------------------------------------
# Create simulator and optimizer
# -------------------------------------------------

stage_simulator = StageSimulator()
simulator = MechanismSimulator(
    motion=motion,
    stage_simulator=stage_simulator,
    stage_limit=None,
)

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

# Population size enlarged for the PPBLS mechanism (4 levers,
# 8 parameters, two coupled stages) to reduce the chance of
# early stagnation through blocking candidates.
POPULATION_SIZE = 400
SELECTION_COUNT = 80
CHILDREN_COUNT = 400

# Adaptive step-size control via the Rechenberg 1/5 success
# rule: the mutation strength is increased when many
# generations improve the best solution (explore further)
# and decreased when few do (exploit the current region).
ADAPTIVE_WINDOW = 10
ADAPTIVE_MIN_STRENGTH = 0.005
ADAPTIVE_MAX_STRENGTH = 0.5

rng = random.Random(42)  # Fixed seed for reproducibility
population_factory = PopulationFactory(random_generator=rng)
population = population_factory.create(parameter_template, size=POPULATION_SIZE)

# Mutation operator is shared with the reproduction and the
# adaptive controller so the controller can update its
# strength in place every ADAPTIVE_WINDOW generations.
mutation_operator = ParameterMutation(
    strength=0.1,
    distribution="gauss",
    random_generator=rng,
)
adaptive_controller = AdaptiveStrength(
    mutation_operator,
    window=ADAPTIVE_WINDOW,
    min_strength=ADAPTIVE_MIN_STRENGTH,
    max_strength=ADAPTIVE_MAX_STRENGTH,
)

# -------------------------------------------------
# Evolution engine
# -------------------------------------------------

engine = EvolutionEngine(
    population=population,
    evaluator=optimizer.evaluate,
    selection_count=SELECTION_COUNT,
    reproduction=Reproduction(
        mutation=mutation_operator,
        # Crossover: recombine two parents per child (grouped by
        # lever) so good sub-configurations of individual levers can
        # be combined.  Each child is the recombination of two
        # randomly chosen survivors followed by Gaussian mutation.
        crossover=Crossover(
            strategy="grouped",
            random_generator=rng,
        ),
    ),
    target_fitness=0.01,
    max_generations=1000,
    stagnation_limit=200,
    stagnation_tolerance=1e-8,
    # Adaptive 1/5 success-rule step-size control.
    adaptive_strength=adaptive_controller,
)

# -------------------------------------------------
# Initial validation
# -------------------------------------------------

print("\n" + "=" * 80)
print("EVOLUTION ENGINE")
print("=" * 80)
print(f"\nPopulation size: {POPULATION_SIZE}")
print(f"Selection count: {SELECTION_COUNT}")
print(f"Children count:  {CHILDREN_COUNT}")

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

for generation in engine.run(children_count=CHILDREN_COUNT):
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

# Final adaptierte Mutationsstaerke (1/5-Erfolgsregel).
print(f"\nAdaptive mutation strength (1/5 success rule):")
print(f"  Initial strength : {adaptive_controller.history[0]:.6f}")
print(f"  Final strength   : {adaptive_controller.strength:.6f}")
print(f"  Adaptations      : {len(adaptive_controller.history) - 1}")
if len(adaptive_controller.history) > 1:
    print(f"  Strength history : " + ", ".join(
        f"{s:.4f}" for s in adaptive_controller.history
    ))

cache_stats = optimizer.get_cache_stats()
print(f"\nCache statistics:")
print(f"  Evaluations: {cache_stats['evaluations']}")
print(f"  Cache hits: {cache_stats['cache_hits']}")
print(f"  Cache misses: {cache_stats['cache_misses']}")
print(f"  Cache size: {cache_stats['cache_size']}")

# -------------------------------------------------
# BESTE HEBELKONFIGURATION (OPTIMIERT) - complete lever
# overview of the best candidate, with the optimized angle
# (set by the optimizer) and the optimized stage rod length
# for physical reconstruction.
# -------------------------------------------------

print("\n" + "=" * 80)
print("BESTE HEBELKONFIGURATION (OPTIMIERT)")
print("=" * 80)

if engine.best_candidate is not None:
    best = engine.best_candidate
    best_values = best.values()

    print("  Optimierter Winkelwert je Hebel (vom Optimierer gesetzter")
    print("  angle-Parameter des besten Kandidaten) und optimierte")
    print("  Stangenlaenge je Stufe.\n")

    # Optimized angle and length per lever, read from best_candidate.
    for lever in definition.levers:
        length_val = best_values.get(f"lever.{lever.id}.length", lever.length_start)
        angle_val = best_values.get(f"lever.{lever.id}.angle", lever.angle_start)
        driver_str = f"driver={lever.driver}" if lever.driver is not None else "-"
        coupled_str = f"coupled={lever.coupled}" if lever.coupled is not None else "-"
        chain = driver_str if lever.driver is not None else coupled_str
        print(f"  Lever {lever.id}:")
        print(f"    Drehpunkt (pivot)       : ({lever.pivot.x:.3f}, {lever.pivot.y:.3f}, {lever.pivot.z:.3f})")
        print(f"    Achse (axis)            : ({lever.axis.x:.3f}, {lever.axis.y:.3f}, {lever.axis.z:.3f})")
        print(f"    Winkelbereich          : [{math.degrees(lever.angle_min):.3f}deg, {math.degrees(lever.angle_max):.3f}deg]")
        print(f"    OPTIMIERTER WINKEL      : {math.degrees(angle_val):.3f}deg")
        print(f"    Laengenbereich         : [{lever.length_min:.3f}, {lever.length_max:.3f}] mm")
        print(f"    OPTIMIERTE LAENGE       : {length_val:.3f} mm")
        print(f"    Verkettung             : {chain}")
        print()

    # Build the best mechanism to read the optimized rod length
    # (automatically computed from the reference positions) per stage.
    mechanism = builder.build(best)

    print("-" * 55)
    print("  OPTIMIERTE STANGENLAENGE JE STUFE")
    print("-" * 55)
    for index, stage in enumerate(mechanism.stages, start=1):
        print(f"  Stage {index}:")
        print(f"    Eingangs-Hebel  : length={stage.input_lever.length:.3f} mm, "
              f"pivot=({stage.input_lever.pivot.x:.3f}, {stage.input_lever.pivot.y:.3f}, {stage.input_lever.pivot.z:.3f})")
        print(f"    Ausgangs-Hebel  : length={stage.output_lever.length:.3f} mm, "
              f"pivot=({stage.output_lever.pivot.x:.3f}, {stage.output_lever.pivot.y:.3f}, {stage.output_lever.pivot.z:.3f})")
        print(f"    Stangenlaenge  : {stage.rod_length:.3f} mm")
        print(f"    Eingangsbereich: [{math.degrees(stage.input_angle_min):.3f}deg, {math.degrees(stage.input_angle_max):.3f}deg]")
        print(f"    Ausgangsbereich: [{math.degrees(stage.output_angle_min):.3f}deg, {math.degrees(stage.output_angle_max):.3f}deg]")
        print()

    print("=" * 80)
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
        print(f"    Input range:  [{math.degrees(stage.input_angle_min):.1f}deg, {math.degrees(stage.input_angle_max):.1f}deg]")
        print(f"    Output range: [{math.degrees(stage.output_angle_min):.1f}deg, {math.degrees(stage.output_angle_max):.1f}deg]")

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
                print(f"    Failed at: {math.degrees(result.failed_at_input_angle):.1f}deg")

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
                print(f"    Output range: [{min(output_deg):.1f}deg, {max(output_deg):.1f}deg]")
        else:
            if stage_result.blocked_at is not None:
                print(f"    Blocked at: {math.degrees(stage_result.blocked_at):.1f}deg")

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
else:
    print("  Kein gueltiger bester Kandidat gefunden.")
