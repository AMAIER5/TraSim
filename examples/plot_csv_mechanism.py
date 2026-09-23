"""examples/plot_csv_mechanism.py

Visualization of an optimization result as an HTML
document.

The script loads a mechanism from mechanism.csv,
optimizes it against target_curve.csv with the
evolutionary optimizer, stores the optimized
mechanism in mechanism_optimized.csv and writes an
HTML document (simulation_result.html) with two
diagrams:

1. A three-dimensional diagram showing every lever
   of the OPTIMIZED mechanism as a line in three
   positions: at the minimum, the middle and the
   maximum drive angle of the first (driving) lever.
   The coupling rods between the levers are drawn as
   well.
2. A two-dimensional diagram showing the desired
   output curve (Soll) and the achieved output curve
   (Ist) over the drive angle.

CSV conventions: input files may use the
international convention (comma, dot) or the German
convention (semicolon, comma as decimal separator);
the convention is detected from the first line.
When any input file uses the German convention, all
files written by this script (mechanism_optimized.csv
and the HTML document) follow the German convention.

User I/O angles are in DEGREES.
Internal calculations use RADIANS.

For stable optimization:

- Use mutation strength of 0.01-0.05 (not 0.15)
- Use sufficient generations (500+)
- Use tight stagnation tolerance (1e-8)
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import replace
from pathlib import Path

from analysis.curve_fitness import CurveFitness
from analysis.curve_plotter import (
    CurvePlotter,
    mechanism_states_from_results,
)
from analysis.target_curve import TargetCurve
from mechanism_io.csv_convention import (
    GERMAN,
    detect_convention,
)
from mechanism_io.csv_reader import CsvReader
from mechanism_io.csv_writer import CsvWriter
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
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import Reproduction
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.point_motion import PointMotion
from simulation.stage_simulator import (
    StageSimulator,
)

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

MECHANISM_FILE = BASE_DIR / "mechanism.csv"
TARGET_FILE = BASE_DIR / "target_curve.csv"
OPTIMIZED_FILE = (
    BASE_DIR / "mechanism_optimized.csv"
)
OUTPUT_FILE = (
    BASE_DIR / "simulation_result.html"
)

# -------------------------------------------------
# Optimization setup
# -------------------------------------------------

POPULATION_SIZE = 100
CHILDREN_COUNT = 100
SELECTION_COUNT = 30
MUTATION_STRENGTH = 0.01
MAX_GENERATIONS = 200
STAGNATION_LIMIT = 100
STAGNATION_TOLERANCE = 1e-8
TARGET_FITNESS = 0.01

# -------------------------------------------------
# CSV conventions of the input files
# -------------------------------------------------

mechanism_convention = detect_convention(
    MECHANISM_FILE,
)
target_convention = detect_convention(
    TARGET_FILE,
)
output_convention = (
    GERMAN
    if (
        mechanism_convention.is_german
        or target_convention.is_german
    )
    else mechanism_convention
)

# -------------------------------------------------
# Mechanism definition (start values)
# -------------------------------------------------

definition = CsvReader.read_mechanism(
    MECHANISM_FILE,
)
builder = CsvMechanismBuilder(definition)

# -------------------------------------------------
# Target curve
# -------------------------------------------------

target_curve = TargetCurve.from_csv(TARGET_FILE)
with open(
    TARGET_FILE,
    newline="",
    encoding="utf-8",
) as file:
    reader = csv.DictReader(
        file,
        delimiter=target_convention.delimiter,
    )
    target_input_angles_deg = [
        target_convention.parse_float(
            row["input_angle"],
        )
        for row in reader
    ]
target_input_angles_rad = tuple(
    math.radians(angle)
    for angle in target_input_angles_deg
)
target_output_angles_rad = tuple(
    target_curve.evaluate(angle)
    for angle in target_input_angles_rad
)

# -------------------------------------------------
# Simulation setup
# -------------------------------------------------

motion = PointMotion(
    angles=target_input_angles_rad,
)
simulator = MechanismSimulator(
    motion=motion,
    stage_simulator=StageSimulator(),
    stage_limit=None,
)
fitness = CurveFitness(
    target_curve=target_curve,
    motion_start=target_input_angles_rad[0],
    motion_range=(
        target_input_angles_rad[-1]
        - target_input_angles_rad[0]
    ),
)
optimizer = MechanismOptimizer(
    builder=builder,
    simulator=simulator,
    fitness=fitness,
)

# -------------------------------------------------
# Optimization
# -------------------------------------------------

print("=" * 80)
print("OPTIMIZATION")
print("=" * 80)

rng = random.Random(42)
parameter_template = CsvParameterFactory.create(
    definition,
)
population = PopulationFactory(
    random_generator=rng,
).create(
    parameter_template,
    size=POPULATION_SIZE,
)
engine = EvolutionEngine(
    population=population,
    evaluator=optimizer.evaluate,
    selection_count=SELECTION_COUNT,
    reproduction=Reproduction(
        mutation=ParameterMutation(
            strength=MUTATION_STRENGTH,
            random_generator=rng,
        ),
    ),
    target_fitness=TARGET_FITNESS,
    max_generations=MAX_GENERATIONS,
    stagnation_limit=STAGNATION_LIMIT,
    stagnation_tolerance=(
        STAGNATION_TOLERANCE
    ),
)

engine.evaluate_population()
valid = sum(
    score < float("inf")
    for score in engine.scores.values()
)
print(
    f"\nValid initial candidates: "
    f"{valid}/{len(engine.population)}",
)
if valid == 0:
    raise RuntimeError(
        "No valid candidates - mechanism is "
        "infeasible"
    )

print("\n  Generation |         Fitness")
print("-" * 40)
for generation in engine.run(
    children_count=CHILDREN_COUNT,
):
    print(
        f"  {generation:3d} | "
        f"{engine.best_score:>16.8f}",
    )

best = engine.best_candidate
if best is None:
    raise RuntimeError(
        "Optimization produced no candidate."
    )

print(f"\nStop reason: {engine.stop_reason}")
print(f"Best fitness: {engine.best_score:.12f}")

# -------------------------------------------------
# Optimized mechanism
# -------------------------------------------------

best_values = best.values()
optimized_levers = tuple(
    replace(
        lever,
        length_start=best_values.get(
            f"lever.{lever.id}.length",
            lever.length_start,
        ),
        angle_start=best_values.get(
            f"lever.{lever.id}.angle",
            lever.angle_start,
        ),
    )
    for lever in definition.levers
)


optimized_definition = replace(
    definition,
    levers=optimized_levers,
)
CsvWriter.write_mechanism(
    optimized_definition,
    OPTIMIZED_FILE,
    convention=output_convention,
)
print(
    f"\nOptimized mechanism written to: "
    f"{OPTIMIZED_FILE}",
)

optimized_mechanism = builder.build(best)

# -------------------------------------------------
# Simulation of the optimized mechanism
# -------------------------------------------------

simulation_results = simulator.simulate(
    optimized_mechanism,
)

final_result = simulation_results[-1]
drive_result = simulation_results[0]
all_succeeded = all(
    result.success
    for result in simulation_results
)
if not all_succeeded:
    blocked = next(
        (
            result.blocked_at
            for result in simulation_results
            if not result.success
        ),
        None,
    )
    raise RuntimeError(
        "Simulation blocked at "
        f"{math.degrees(blocked):.1f} deg; "
        "no complete result to plot."
    )

# -------------------------------------------------
# Plot data (optimized mechanism)
# -------------------------------------------------

plotter = CurvePlotter(
    title=(
        "TraSim \u2014 Optimiertes Getriebe "
        f"({OPTIMIZED_FILE.name})"
    ),
)

plotter.set_target_curve(
    target_curve,
    input_angles=target_input_angles_rad,
    output_angles=target_output_angles_rad,
)

plotter.add_actual_curve(
    drive_result.input_angles,
    final_result.output_angles,
    label="Ist-Ausgangskurve",
)

for state in mechanism_states_from_results(
    optimized_mechanism,
    simulation_results,
):
    plotter.add_mechanism_state(state)

# -------------------------------------------------
# Write HTML document
# -------------------------------------------------

output_path = plotter.write(OUTPUT_FILE)

print("=" * 80)
print("VISUALIZATION")
print("=" * 80)
print(
    f"\nDrive angles : {len(drive_result.input_angles)} points",
)
print(
    f"Output curve : {len(final_result.output_angles)} points",
)
print(f"\nHTML document written to: {output_path}")
