"""
examples/optimize_csv_mechanism_Step3.py

Schritt 3: Zusätzliche Getriebestufe, die nie blockieren kann anhaengen

User I/O angles are in DEGREES.
Internal calculations use RADIANS.

Dieses Beispiel verwendet einen Mechanismus mit DREI Hebeln:
- Hebel 1 (Eingang):  Drehpunkt bei (0,0,0),   Laenge 100mm, Winkelbereich [-50, 50]
- Hebel 2 (Zwischen): Drehpunkt bei (100,0,0), Laenge 100mm, Winkelbereich [-50, 50], angetrieben von Hebel 1
- Hebel 3 (Ausgang): Drehpunkt bei (200,0,0), Laenge 100mm, Winkelbereich [-50, 50], angetrieben von Hebel 2

Die Zielkurve ist eine einfache 1:1 Uebersetzung von -45 bis +45.

Hinweis: Beide Stufen (1->2 und 2->3) bilden Rhomben mit 100mm Abstand
zwischen den Drehpunkten. Da jede Stufe einzeln nie blockieren kann,
kan der gesamte Mechanismus auch nie blockieren.

Vergleich mit Schritt 1:
  Schritt 1: 1 Stufe (2 Hebel) -> nie blockierend
  Schritt 3: 2 Stufen (3 Hebel) -> nie blockierend (beide Stufen sind Rhomben)
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
# Paths - Schritt 3
# -------------------------------------------------

BASE_DIR = Path(__file__).parent.absolute() if '__file__' in globals() else Path.cwd() / "examples"

MECHANISM_FILE = BASE_DIR / "mechanism_Step3.csv"
TARGET_FILE = BASE_DIR / "target_curve_Step3.csv"

# -------------------------------------------------
# Mechanism definition
# -------------------------------------------------

definition = CsvReader.read_mechanism(MECHANISM_FILE)
builder = CsvMechanismBuilder(definition)

# -------------------------------------------------
# Print mechanism info
# -------------------------------------------------

print("=" * 80)
print("SCHRITT 3: ZUSAETZLICHE NICHT-BLOCKIERENDE GETRIEBESTUFE")
print("=" * 80)
print(f"\nMechanismus: {len(definition.levers)} Hebel")
for lever in definition.levers:
    driver_str = f", driver={lever.driver}" if lever.driver else ""
    coupled_str = f", coupled={lever.coupled}" if lever.coupled else ""
    print(f"  Hebel {lever.id}: pivot=({lever.pivot.x:.1f}, {lever.pivot.y:.1f}, {lever.pivot.z:.1f}), "
          f"axis=({lever.axis.x:.1f}, {lever.axis.y:.1f}, {lever.axis.z:.1f}), "
          f"length={lever.length_start:.1f}mm, "
          f"angle=[{math.degrees(lever.angle_min):.1f}, {math.degrees(lever.angle_max):.1f}]"
          f"{driver_str}{coupled_str}")

print("\nHinweis: Beide Stufen bilden Rhomben (100mm Hebel, 100mm Abstand).")
print("         Der gesamte Mechanismus kann daher nie blockieren.")

# -------------------------------------------------
# Target curve - 1:1 Uebersetzung von -45 bis +45
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
print("ZIELKURVE (1:1 Uebersetzung ueber 90 Grad Drehwinkel)")
print("=" * 80)
print(f"\nZielkurve: {len(target_input_angles_deg)} Punkte")
for inp, out in zip(target_input_angles_deg, target_output_angles_deg):
    print(f"  {inp:.1f} -> {out:.1f}")

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
print("OPTIMIERUNGS-EINSTELLUNGEN")
print("=" * 80)
print(f"\nParameter ({len(definition.levers) * 2}):")

parameter_template = CsvParameterFactory.create(definition)

for param in parameter_template.parameters:
    if "angle" in param.name:
        min_deg = math.degrees(param.minimum)
        max_deg = math.degrees(param.maximum)
        val_deg = math.degrees(param.value)
        print(f"  {param.name}: [{min_deg:.1f}, {max_deg:.1f}], default={val_deg:.1f}")
    else:
        print(f"  {param.name}: [{param.minimum:.1f}, {param.maximum:.1f}], default={param.value:.1f}")

print(f"\nBewegungsbereich: {math.degrees(min_input_rad):.1f} bis {math.degrees(max_input_rad):.1f}, "
      f"Schritt=2.0")

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

rng = random.Random(42)  # Fixed seed for reproducibility
population_factory = PopulationFactory(random_generator=rng)
population = population_factory.create(parameter_template, size=50)

# -------------------------------------------------
# Evolution engine
# -------------------------------------------------

engine = EvolutionEngine(
    population=population,
    evaluator=optimizer.evaluate,
    selection_count=15,
    reproduction=Reproduction(
        mutation=ParameterMutation(
            strength=0.01,
            random_generator=rng,
        ),
    ),
    target_fitness=0.02,
    max_generations=500,
    stagnation_limit=100,
    stagnation_tolerance=1e-8,
)

# -------------------------------------------------
# Initial validation
# -------------------------------------------------

print("\n" + "=" * 80)
print("EVOLUTIONS-ENGINE")
print("=" * 80)

engine.evaluate_population()
valid = sum(score < float("inf") for score in engine.scores.values())

print("\n" + "=" * 80)
print("INITIALE POPULATION")
print("=" * 80)
print(f"Gueltige Kandidaten: {valid}/{len(engine.population)}")
if engine.best_score < float("inf"):
    print(f"Beste anfaengliche Fitness: {engine.best_score:.8f}")
else:
    print("Beste anfaengliche Fitness: inf (keine gueltigen Kandidaten)")

if valid == 0:
    raise RuntimeError("Keine gueltigen Kandidaten - Mechanismus ist nicht machbar")

# -------------------------------------------------
# Evolution loop
# -------------------------------------------------

print("\n" + "=" * 80)
print("OPTIMIERUNGS-FORTSCHRITT")
print("=" * 80)
print("  Generation |         Fitness |  Verbesserung")
print("-" * 55)

prev_best = float("inf")
generation_count = 0

for generation in engine.run(children_count=50):
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
print("OPTIMIERUNGS-ERGEBNISSE")
print("=" * 80)

print(f"\nStoppgrund: {engine.stop_reason}")
print(f"Beste Fitness: {engine.best_score:.12f}")
print(f"Generationen durchlaufen: {generation_count}")

cache_stats = optimizer.get_cache_stats()
print(f"\nCache-Statistik:")
print(f"  Bewertungen: {cache_stats['evaluations']}")
print(f"  Cache-Treffer: {cache_stats['cache_hits']}")
print(f"  Cache-Fehlschlaege: {cache_stats['cache_misses']}")
print(f"  Cache-Groesse: {cache_stats['cache_size']}")

# -------------------------------------------------
# Best mechanism details
# -------------------------------------------------

if engine.best_candidate is not None:
    mechanism = builder.build(engine.best_candidate)

    print("\n" + "=" * 80)
    print("BESTER MECHANISMUS")
    print("=" * 80)
    print(f"\nAnzahl Stufen: {len(mechanism.stages)}")
    
    for index, stage in enumerate(mechanism.stages, start=1):
        print(f"\n  Stufe {index}:")
        print(f"    Eingangshebel:  Laenge={stage.input_lever.length:.2f} mm, "
              f"Drehpunkt=({stage.input_lever.pivot.x:.1f}, {stage.input_lever.pivot.y:.1f}, {stage.input_lever.pivot.z:.1f})")
        print(f"    Ausgangshebel: Laenge={stage.output_lever.length:.2f} mm, "
              f"Drehpunkt=({stage.output_lever.pivot.x:.1f}, {stage.output_lever.pivot.y:.1f}, {stage.output_lever.pivot.z:.1f})")
        print(f"    Stangenlaenge:  {stage.rod_length:.2f} mm")
        print(f"    Eingangsbereich:  [{math.degrees(stage.input_angle_min):.1f}, {math.degrees(stage.input_angle_max):.1f}]")
        print(f"    Ausgangsbereich: [{math.degrees(stage.output_angle_min):.1f}, {math.degrees(stage.output_angle_max):.1f}]")
        
        # Check if this stage is a rhombus
        pivot_dx = stage.output_lever.pivot.x - stage.input_lever.pivot.x
        pivot_dy = stage.output_lever.pivot.y - stage.input_lever.pivot.y
        pivot_dist = math.sqrt(pivot_dx**2 + pivot_dy**2)
        is_rhombus = abs(stage.rod_length - pivot_dist) < 0.01 and abs(stage.input_lever.length - stage.output_lever.length) < 0.01
        print(f"    Rhombus: {is_rhombus} (pivot_dist={pivot_dist:.2f}, rod={stage.rod_length:.2f})")

    # Validation results
    validation_results = builder.get_validation_results()
    print("\n" + "-" * 55)
    print("STUFEN-VALIDIERUNGSERGEBNISSE")
    print("-" * 55)
    for stage_idx, result in enumerate(validation_results, start=1):
        print(f"\n  Stufe {stage_idx}:")
        print(f"    Gueltig: {result.valid}")
        if not result.valid:
            print(f"    Grund: {result.reason}")
            if hasattr(result, 'failed_at_input_angle') and result.failed_at_input_angle is not None:
                print(f"    Fehlgeschlagen bei: {math.degrees(result.failed_at_input_angle):.1f}")

    # Simulation results
    print("\n" + "-" * 55)
    print("SIMULATIONSERGEBNISSE")
    print("-" * 55)

    simulation_results = simulator.simulate(mechanism)

    for stage_index, stage_result in enumerate(simulation_results, start=1):
        print(f"\n  Stufe {stage_index}:")
        print(f"    Erfolg: {stage_result.success}")
        print(f"    Eingabepunkte: {len(stage_result.input_angles)}")
        if stage_result.success:
            print(f"    Ausgabepunkte: {len(stage_result.output_angles)}")
            if stage_result.output_angles:
                output_deg = tuple(math.degrees(a) for a in stage_result.output_angles)
                print(f"    Ausgangsbereich: [{min(output_deg):.1f}, {max(output_deg):.1f}]")
        else:
            if stage_result.blocked_at is not None:
                print(f"    Blockiert bei: {math.degrees(stage_result.blocked_at):.1f}")

    # Solver statistics
    print("\n" + "=" * 80)
    print("SOLVER-STATISTIK")
    print("=" * 80)
    total_stats = {}
    for solver in stage_simulator.solvers:
        stats = solver.get_stats()
        for key, value in stats.items():
            total_stats[key] = total_stats.get(key, 0) + value
    for key, value in sorted(total_stats.items()):
        print(f"  {key:>25}: {value:>10}")

print("\n" + "=" * 80)
print("SCHRITT 3 ABGESCHLOSSEN")
print("=" * 80)
print("\nHinweis: Dieser Mechanismus besteht aus ZWEI hintereinander")
print("geschalteten Rhombus-Stufen (jeweils 100mm Hebel, 100mm Abstand).")
print("Da JEDE einzelne Stufe nie blockieren kann, kann auch der")
print("gesamte Mechanismus nie blockieren.")
print("\nVergleich:")
print("  Schritt 1: 1 Stufe (2 Hebel) -> nie blockierend")
print("  Schritt 2: 1 Stufe (2 Hebel) -> blockiert ausserhalb [-45, 45]")
print("  Schritt 3: 2 Stufen (3 Hebel) -> nie blockierend (beide Stufen sind Rhomben)")
print("=" * 80)
