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

Parallel evaluation:
- WORKERS = 1 keeps the sequential mode (default behaviour).
- WORKERS > 1 evaluates every generation via a process pool
  (builder/simulator/fitness are built once per worker process).
  The pool uses order-preserving map, so the result is
  identical to the sequential run.

On Windows the ``if __name__ == "__main__":`` guard is REQUIRED
for the worker processes (multiprocessing spawn re-imports this
module; without the guard every import would start a new run).
"""

from __future__ import annotations

import csv
import math
import os
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
# Parallel evaluation
# -------------------------------------------------
#
# WORKERS = 1 -> sequential (default behaviour, no process pool).
# WORKERS > 1 -> the population of every generation is evaluated
#                by a process pool; builder/simulator/fitness are
#                built once per worker process via the pool
#                initializer.  Order-preserving pool.map keeps the
#                run deterministic.
WORKERS = os.cpu_count() or 1

# Process-local state of a worker (set once by the
# evaluator factory, used by the stats provider).
_WORKER_STAGE_SIMULATOR: StageSimulator | None = None


# -------------------------------------------------
# Worker process helpers
# -------------------------------------------------

def read_target_curve(
    path: Path,
) -> tuple[list[float], list[float], TargetCurve]:
    """
    Read the target curve CSV.

    Returns the support point angles in degrees and the
    non-interpolating TargetCurve built from the same file.
    """

    target_input_angles_deg = []
    target_output_angles_deg = []

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_input_angles_deg.append(float(row['input_angle']))
            target_output_angles_deg.append(float(row['output_angle']))

    target_curve = TargetCurve.from_csv_strict(path)

    return (
        target_input_angles_deg,
        target_output_angles_deg,
        target_curve,
    )


def _worker_evaluator_factory():
    """
    Evaluator factory for the worker processes.

    Builds the complete evaluation stack (CSV definition,
    mechanism builder, simulator, fitness) ONCE per process.
    The returned evaluator is used for every candidate this
    worker evaluates.
    """

    global _WORKER_STAGE_SIMULATOR

    definition = CsvReader.read_mechanism(MECHANISM_FILE)
    builder = CsvMechanismBuilder(definition)

    target_input_angles_deg, _, target_curve = read_target_curve(TARGET_FILE)

    min_input_rad = min(math.radians(a) for a in target_input_angles_deg)
    max_input_rad = max(math.radians(a) for a in target_input_angles_deg)
    travel_range_rad = max_input_rad - min_input_rad

    motion = PointMotion(
        angles=tuple(math.radians(a) for a in target_input_angles_deg),
    )

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

    _WORKER_STAGE_SIMULATOR = stage_simulator

    return optimizer.evaluate


def _worker_solver_stats() -> dict[str, int]:
    """
    Cumulative solver statistics of THIS worker process.

    Sums the stats of every solver created by the
    process-local stage simulator.  The engine collects
    one answer per worker and de-duplicates by pid.
    """

    if _WORKER_STAGE_SIMULATOR is None:
        return {}

    totals: dict[str, int] = {}
    for solver in _WORKER_STAGE_SIMULATOR.solvers:
        for key, value in solver.get_stats().items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _print_mechanism_info(definition) -> None:
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


def _print_lever_overview(definition) -> None:
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


def _print_target_curve(
    target_input_angles_deg,
    target_output_angles_deg,
) -> None:
    print("\n" + "=" * 80)
    print("LOADING TARGET CURVE")
    print("=" * 80)
    print(f"\nTarget curve: {len(target_input_angles_deg)} points")
    for inp, out in zip(target_input_angles_deg, target_output_angles_deg):
        print(f"  {inp:.1f}deg -> {out:.1f}deg")


def _print_simulation_setup(
    definition,
    target_input_angles_deg,
    parameter_template,
) -> None:
    min_input_rad = min(math.radians(a) for a in target_input_angles_deg)
    max_input_rad = max(math.radians(a) for a in target_input_angles_deg)

    print("\n" + "=" * 80)
    print("OPTIMIZATION SETUP")
    print("=" * 80)
    print(f"\nParameters ({len(definition.levers) * 2}):")

    for param in parameter_template.parameters:
        if "angle" in param.name:
            min_deg = math.degrees(param.minimum)
            max_deg = math.degrees(param.maximum)
            val_deg = math.degrees(param.value)
            print(f"  {param.name}: [{min_deg:.1f}deg, {max_deg:.1f}deg], default={val_deg:.1f}deg")
        else:
            print(f"  {param.name}: [{param.minimum:.1f}, {param.maximum:.1f}], default={param.value:.1f}")

    print(f"\nMotion: simulated at the {len(target_input_angles_deg)} prescribed "
          f"support points ({math.degrees(min_input_rad):.1f}deg to "
          f"{math.degrees(max_input_rad):.1f}deg), no interpolation")


def _print_best_configuration(
    definition,
    builder,
    best: ParameterSet,
) -> None:
    print("\n" + "=" * 80)
    print("BESTE HEBELKONFIGURATION (OPTIMIERT)")
    print("=" * 80)

    best_values = best.values()

    print("  Optimierter Winkelwert je Hebel (vom Optimierer gesetzter")
    print("  angle-Parameter des besten Kandidaten) und optimierte")
    print("  Stangenlaenge je Stufe.\n")

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

    return mechanism


def _print_validation_and_simulation(
    builder,
    simulator,
    mechanism,
    target_input_angles_deg,
    target_output_angles_deg,
) -> None:
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

    print("\n" + "-" * 55)
    print("UEBERTRAGUNGSKURVE (erreichte Winkel)")
    print("-" * 55)

    final_result = simulation_results[-1]
    all_succeeded = all(r.success for r in simulation_results)
    if (
        all_succeeded
        and len(final_result.output_angles) == len(target_input_angles_deg)
    ):
        print("input_angle,output_angle_target,output_angle_final")
        for inp_deg, tgt_deg, out_rad in zip(
            target_input_angles_deg,
            target_output_angles_deg,
            final_result.output_angles,
        ):
            out_deg = math.degrees(out_rad)
            print(f"{inp_deg:g},{tgt_deg:.1f},{round(out_deg)}")
    else:
        print("  Simulation blockiert - keine vollstaendige Uebertragungskurve.")


def _print_solver_statistics(
    stage_simulator,
    engine,
) -> None:
    print("\n" + "=" * 80)
    print("SOLVER STATISTICS")
    print("=" * 80)

    total_stats: dict[str, int] = {}

    # Main-process stats: in sequential mode these cover every
    # evaluation; in parallel mode the main process performs no
    # evaluations and its stats stay empty.
    for solver in stage_simulator.solvers:
        stats = solver.get_stats()
        for key, value in stats.items():
            total_stats[key] = total_stats.get(key, 0) + value

    # Worker stats: aggregated back from every worker process.
    worker_stats = engine.collect_worker_stats()
    for key, value in worker_stats.items():
        total_stats[key] = total_stats.get(key, 0) + value

    if WORKERS > 1:
        print(f"  (main process: {WORKERS} worker processes, stats aggregated)")

    for key, value in sorted(total_stats.items()):
        print(f"  {key:>25}: {value:>10}")


def main() -> None:

    # -------------------------------------------------
    # Mechanism definition
    # -------------------------------------------------

    definition = CsvReader.read_mechanism(MECHANISM_FILE)
    builder = CsvMechanismBuilder(definition)

    # -------------------------------------------------
    # Print mechanism info
    # -------------------------------------------------

    _print_mechanism_info(definition)
    _print_lever_overview(definition)

    # -------------------------------------------------
    # Target curve - READ DIRECTLY FROM CSV
    # -------------------------------------------------

    target_input_angles_deg, target_output_angles_deg, target_curve = (
        read_target_curve(TARGET_FILE)
    )

    target_input_angles_rad = tuple(math.radians(a) for a in target_input_angles_deg)

    # Non-interpolating target curve: fitness is only defined at the
    # 11 prescribed support points; no linear interpolation between
    # them is used for evaluation.

    _print_target_curve(
        target_input_angles_deg,
        target_output_angles_deg,
    )

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

    parameter_template = CsvParameterFactory.create(definition)

    _print_simulation_setup(
        definition,
        target_input_angles_deg,
        parameter_template,
    )

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
        # In parallel mode the main-process evaluator is unused;
        # every worker builds its own evaluation stack once via
        # the factory below (see EvolutionEngine._worker_initializer).
        evaluator=optimizer.evaluate if WORKERS == 1 else None,
        selection_count=SELECTION_COUNT,
        reproduction=Reproduction(
            mutation=mutation_operator,
            # Seeded parent selection keeps the whole run
            # reproducible (independent of WORKERS).
            random_generator=rng,
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
        # Parallel evaluation (WORKERS = 1 -> sequential default).
        workers=WORKERS,
        evaluator_factory=(
            _worker_evaluator_factory
            if WORKERS > 1
            else None
        ),
        stats_provider=(
            _worker_solver_stats
            if WORKERS > 1
            else None
        ),
    )

    try:
        # -------------------------------------------------
        # Initial validation
        # -------------------------------------------------

        print("\n" + "=" * 80)
        print("EVOLUTION ENGINE")
        print("=" * 80)
        print(f"\nPopulation size: {POPULATION_SIZE}")
        print(f"Selection count: {SELECTION_COUNT}")
        print(f"Children count:  {CHILDREN_COUNT}")
        print(f"Workers:         {WORKERS}"
              + ("  (sequential)" if WORKERS == 1 else "  (parallel)"))

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
        if WORKERS > 1:
            print("  (main process only - every worker keeps its own cache)")

        # -------------------------------------------------
        # BESTE HEBELKONFIGURATION (OPTIMIERT)
        # -------------------------------------------------

        if engine.best_candidate is not None:
            mechanism = _print_best_configuration(
                definition,
                builder,
                engine.best_candidate,
            )

            _print_validation_and_simulation(
                builder,
                simulator,
                mechanism,
                target_input_angles_deg,
                target_output_angles_deg,
            )

            _print_solver_statistics(
                stage_simulator,
                engine,
            )
        else:
            print("  Kein gueltiger bester Kandidat gefunden.")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
