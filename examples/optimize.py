"""
examples/optimize.py

Combined end-to-end optimization and visualization.

Combines the terminal output and parallel
(multi-core) evaluation of optimize_ppbls.py with
the HTML visualization of plot_csv_mechanism.py:

1. Loads a mechanism from mechanism.csv and a
   target curve from target_curve.csv.
2. Optimizes the mechanism with the evolutionary
   optimizer (multiple worker processes).
3. Prints the complete report on the terminal:
   mechanism overview, target curve, optimization
   setup, evolution progress, best configuration,
   stage validation, simulation results, transfer
   curve and solver statistics.
4. Stores the optimized mechanism in
   mechanism_optimized.csv and writes an HTML
   document (simulation_result.html) with two
   diagrams:
   - a three-dimensional diagram showing every
     lever of the OPTIMIZED mechanism as a line in
     three positions (minimum / middle / maximum
     drive angle of the first lever) including the
     coupling rods,
   - a two-dimensional diagram showing the desired
     output curve (Soll) and the achieved output
     curve (Ist) over the drive angle.

CSV conventions: input files may use the
international convention (comma, dot) or the German
convention (semicolon, comma as decimal separator);
the convention is detected from the first line.
When any input file uses the German convention, all
files written by this script follow the German
convention.

User I/O angles are in DEGREES.
Internal calculations use RADIANS.

For stable optimization:

- Use mutation strength of 0.01-0.05 (not 0.15)
- Use sufficient generations (500+)
- Use tight stagnation tolerance (1e-8)

Parallel evaluation:

- WORKERS = 1 keeps the sequential mode.
- WORKERS > 1 evaluates every generation via a
  process pool (builder/simulator/fitness are
  built once per worker process).  The pool uses
  order-preserving map, so the result is identical
  to the sequential run.

On Windows the ``if __name__ == "__main__":`` guard
is REQUIRED for the worker processes.
"""

from __future__ import annotations

import csv
import math
import os
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
from optimization.adaptive_strength import (
    AdaptiveStrength,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.crossover import Crossover
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
# Parallel evaluation
# -------------------------------------------------

WORKERS = os.cpu_count() or 1

# -------------------------------------------------
# Optimization setup
# -------------------------------------------------

POPULATION_SIZE = 400
CHILDREN_COUNT = 400
SELECTION_COUNT = 80
ADAPTIVE_WINDOW = 10
ADAPTIVE_MIN_STRENGTH = 0.005
ADAPTIVE_MAX_STRENGTH = 0.5
TARGET_FITNESS = 0.01
MAX_GENERATIONS = 1000
STAGNATION_LIMIT = 200
STAGNATION_TOLERANCE = 1e-8

# Process-local state of a worker (set once by the
# evaluator factory, used by the stats provider).
_WORKER_STAGE_SIMULATOR: (
    StageSimulator | None
) = None


def read_target_curve(
    path: Path,
) -> tuple[list[float], list[float]]:
    """
    Read the target curve CSV with the detected
    convention and return the support point angles
    in degrees.
    """

    convention = detect_convention(path)
    input_angles_deg: list[float] = []
    output_angles_deg: list[float] = []

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file,
            delimiter=convention.delimiter,
        )
        for row in reader:
            input_angles_deg.append(
                convention.parse_float(
                    row["input_angle"],
                ),
            )
            output_angles_deg.append(
                convention.parse_float(
                    row["output_angle"],
                ),
            )

    return input_angles_deg, output_angles_deg


def _worker_evaluator_factory():
    """
    Evaluator factory for the worker processes.

    Builds the complete evaluation stack (CSV
    definition, mechanism builder, simulator,
    fitness) ONCE per process.  The returned
    evaluator is used for every candidate this
    worker evaluates.
    """

    global _WORKER_STAGE_SIMULATOR

    definition = CsvReader.read_mechanism(
        MECHANISM_FILE,
    )
    builder = CsvMechanismBuilder(definition)

    target_input_deg, _ = read_target_curve(
        TARGET_FILE,
    )
    target_curve = TargetCurve.from_csv(
        TARGET_FILE,
    )

    input_rad = tuple(
        math.radians(angle)
        for angle in target_input_deg
    )

    stage_simulator = StageSimulator()
    simulator = MechanismSimulator(
        motion=PointMotion(angles=input_rad),
        stage_simulator=stage_simulator,
        stage_limit=None,
    )

    fitness = CurveFitness(
        target_curve=target_curve,
        motion_start=min(input_rad),
        motion_range=max(input_rad) - min(input_rad),
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
    Cumulative solver statistics of THIS worker
    process.
    """

    if _WORKER_STAGE_SIMULATOR is None:
        return {}

    totals: dict[str, int] = {}
    for solver in (
        _WORKER_STAGE_SIMULATOR.solvers
    ):
        for key, value in (
            solver.get_stats().items()
        ):
            totals[key] = totals.get(key, 0) + value
    return totals


def _print_mechanism_info(definition) -> None:
    print("=" * 80)
    print("LOADING MECHANISM")
    print("=" * 80)
    print(f"\nMechanism: {len(definition.levers)} levers")
    for lever in definition.levers:
        driver_str = (
            f", driver={lever.driver}"
            if lever.driver
            else ""
        )
        coupled_str = (
            f", coupled={lever.coupled}"
            if lever.coupled
            else ""
        )
        print(
            f"  Lever {lever.id}: "
            f"pivot=({lever.pivot.x:.1f}, "
            f"{lever.pivot.y:.1f}, "
            f"{lever.pivot.z:.1f}), "
            f"axis=({lever.axis.x:.1f}, "
            f"{lever.axis.y:.1f}, "
            f"{lever.axis.z:.1f}), "
            f"length={lever.length_start:.1f}mm, "
            f"angle=["
            f"{math.degrees(lever.angle_min):.1f}deg, "
            f"{math.degrees(lever.angle_max):.1f}deg]"
            f"{driver_str}{coupled_str}",
        )


def _print_lever_overview(definition) -> None:
    print("\n" + "=" * 80)
    print("HEBEL-UEBERSICHT (INITIAL)")
    print("=" * 80)
    print(
        "  Vollstaendige Uebersicht aller Hebel aus"
        " der Mechanismusdefinition.",
    )
    print(
        "  angle_start ist der initiale Winkel JESES"
        " Hebels in Grad.\n",
    )

    for lever in definition.levers:
        driver_str = (
            f"driver={lever.driver}"
            if lever.driver is not None
            else "-"
        )
        coupled_str = (
            f"coupled={lever.coupled}"
            if lever.coupled is not None
            else "-"
        )
        chain = (
            driver_str
            if lever.driver is not None
            else coupled_str
        )
        print(f"  Lever {lever.id}:")
        print(
            f"    Drehpunkt (pivot) : "
            f"({lever.pivot.x:.3f}, "
            f"{lever.pivot.y:.3f}, "
            f"{lever.pivot.z:.3f})",
        )
        print(
            f"    Achse (axis)      : "
            f"({lever.axis.x:.3f}, "
            f"{lever.axis.y:.3f}, "
            f"{lever.axis.z:.3f})",
        )
        print(
            f"    Winkelbereich      : ["
            f"{math.degrees(lever.angle_min):.3f}deg, "
            f"{math.degrees(lever.angle_max):.3f}deg]",
        )
        print(
            f"    INITIALER WINKEL   : "
            f"{math.degrees(lever.angle_start):.3f}deg"
            "   (angle_start)",
        )
        print(
            f"    Laengenbereich     : ["
            f"{lever.length_min:.3f}, "
            f"{lever.length_max:.3f}] mm",
        )
        print(
            f"    length_start       : "
            f"{lever.length_start:.3f} mm",
        )
        print(f"    Verkettung         : {chain}")
        print()


def _print_target_curve(
    target_input_angles_deg,
    target_output_angles_deg,
) -> None:
    print("\n" + "=" * 80)
    print("LOADING TARGET CURVE")
    print("=" * 80)
    print(
        f"\nTarget curve: "
        f"{len(target_input_angles_deg)} points",
    )
    for inp, out in zip(
        target_input_angles_deg,
        target_output_angles_deg,
    ):
        print(f"  {inp:.1f}deg -> {out:.1f}deg")


def _print_simulation_setup(
    definition,
    target_input_angles_deg,
    parameter_template,
) -> None:
    print("\n" + "=" * 80)
    print("OPTIMIZATION SETUP")
    print("=" * 80)
    print(
        f"\nParameters "
        f"({len(definition.levers) * 2}):",
    )

    for param in parameter_template.parameters:
        if "angle" in param.name:
            min_deg = math.degrees(param.minimum)
            max_deg = math.degrees(param.maximum)
            val_deg = math.degrees(param.value)
            print(
                f"  {param.name}: "
                f"[{min_deg:.1f}deg, {max_deg:.1f}deg], "
                f"default={val_deg:.1f}deg",
            )
        else:
            print(
                f"  {param.name}: "
                f"[{param.minimum:.1f}, "
                f"{param.maximum:.1f}], "
                f"default={param.value:.1f}",
            )

    print(
        f"\nMotion: simulated at the "
        f"{len(target_input_angles_deg)} prescribed "
        f"support points, no interpolation",
    )


def _print_best_configuration(
    definition,
    builder,
    best,
) -> None:
    print("\n" + "=" * 80)
    print("BESTE HEBELKONFIGURATION (OPTIMIERT)")
    print("=" * 80)

    best_values = best.values()

    for lever in definition.levers:
        length_val = best_values.get(
            f"lever.{lever.id}.length",
            lever.length_start,
        )
        angle_val = best_values.get(
            f"lever.{lever.id}.angle",
            lever.angle_start,
        )
        driver_str = (
            f"driver={lever.driver}"
            if lever.driver is not None
            else "-"
        )
        coupled_str = (
            f"coupled={lever.coupled}"
            if lever.coupled is not None
            else "-"
        )
        chain = (
            driver_str
            if lever.driver is not None
            else coupled_str
        )
        print(f"  Lever {lever.id}:")
        print(
            f"    Drehpunkt (pivot)       : "
            f"({lever.pivot.x:.3f}, "
            f"{lever.pivot.y:.3f}, "
            f"{lever.pivot.z:.3f})",
        )
        print(
            f"    Achse (axis)            : "
            f"({lever.axis.x:.3f}, "
            f"{lever.axis.y:.3f}, "
            f"{lever.axis.z:.3f})",
        )
        print(
            f"    Winkelbereich          : ["
            f"{math.degrees(lever.angle_min):.3f}deg, "
            f"{math.degrees(lever.angle_max):.3f}deg]",
        )
        print(
            f"    OPTIMIERTER WINKEL      : "
            f"{math.degrees(angle_val):.3f}deg",
        )
        print(
            f"    Laengenbereich         : ["
            f"{lever.length_min:.3f}, "
            f"{lever.length_max:.3f}] mm",
        )
        print(
            f"    OPTIMIERTE LAENGE       : "
            f"{length_val:.3f} mm",
        )
        print(f"    Verkettung             : {chain}")
        print()

    mechanism = builder.build(best)

    print("-" * 55)
    print("  OPTIMIERTE STANGENLAENGE JE STUFE")
    print("-" * 55)
    for index, stage in enumerate(
        mechanism.stages,
        start=1,
    ):
        print(f"  Stage {index}:")
        print(
            f"    Eingangs-Hebel  : "
            f"length={stage.input_lever.length:.3f} mm, "
            f"pivot=({stage.input_lever.pivot.x:.3f}, "
            f"{stage.input_lever.pivot.y:.3f}, "
            f"{stage.input_lever.pivot.z:.3f})",
        )
        print(
            f"    Ausgangs-Hebel  : "
            f"length={stage.output_lever.length:.3f} mm, "
            f"pivot=({stage.output_lever.pivot.x:.3f}, "
            f"{stage.output_lever.pivot.y:.3f}, "
            f"{stage.output_lever.pivot.z:.3f})",
        )
        print(
            f"    Stangenlaenge  : "
            f"{stage.rod_length:.3f} mm",
        )
        print(
            f"    Eingangsbereich: ["
            f"{math.degrees(stage.input_angle_min):.3f}deg, "
            f"{math.degrees(stage.input_angle_max):.3f}deg]",
        )
        print(
            f"    Ausgangsbereich: ["
            f"{math.degrees(stage.output_angle_min):.3f}deg, "
            f"{math.degrees(stage.output_angle_max):.3f}deg]",
        )
        print()

    return mechanism


def _print_validation_and_simulation(
    builder,
    simulator,
    mechanism,
    target_input_angles_deg,
    target_output_angles_deg,
) -> tuple:
    validation_results = (
        builder.get_validation_results()
    )
    print("\n" + "-" * 55)
    print("STAGE VALIDATION RESULTS")
    print("-" * 55)
    for stage_idx, result in enumerate(
        validation_results,
        start=1,
    ):
        print(f"\n  Stage {stage_idx}:")
        print(f"    Valid: {result.valid}")
        if not result.valid:
            print(f"    Reason: {result.reason}")

    print("\n" + "-" * 55)
    print("SIMULATION RESULTS")
    print("-" * 55)

    simulation_results = simulator.simulate(
        mechanism,
    )

    for stage_index, stage_result in enumerate(
        simulation_results,
        start=1,
    ):
        print(f"\n  Stage {stage_index}:")
        print(f"    Success: {stage_result.success}")
        print(
            f"    Input points: "
            f"{len(stage_result.input_angles)}",
        )
        if stage_result.success:
            print(
                f"    Output points: "
                f"{len(stage_result.output_angles)}",
            )
            if stage_result.output_angles:
                output_deg = tuple(
                    math.degrees(angle)
                    for angle in (
                        stage_result.output_angles
                    )
                )
                print(
                    f"    Output range: "
                    f"[{min(output_deg):.1f}deg, "
                    f"{max(output_deg):.1f}deg]",
                )
        else:
            if (
                stage_result.blocked_at
                is not None
            ):
                print(
                    f"    Blocked at: "
                    f"{math.degrees(stage_result.blocked_at):.1f}deg",
                )

    print("\n" + "-" * 55)
    print("UEBERTRAGUNGSKURVE (erreichte Winkel)")
    print("-" * 55)

    final_result = simulation_results[-1]
    all_succeeded = all(
        result.success
        for result in simulation_results
    )
    if (
        all_succeeded
        and len(final_result.output_angles)
        == len(target_input_angles_deg)
    ):
        print(
            "input_angle,output_angle_target,"
            "output_angle_final",
        )
        for inp_deg, tgt_deg, out_rad in zip(
            target_input_angles_deg,
            target_output_angles_deg,
            final_result.output_angles,
        ):
            out_deg = math.degrees(out_rad)
            print(
                f"{inp_deg:g},{tgt_deg:.1f},"
                f"{round(out_deg)}",
            )
    else:
        print(
            "  Simulation blockiert - keine "
            "vollstaendige Uebertragungskurve.",
        )

    return simulation_results


def _print_solver_statistics(
    stage_simulator,
    engine,
) -> None:
    print("\n" + "=" * 80)
    print("SOLVER STATISTICS")
    print("=" * 80)

    total_stats: dict[str, int] = {}

    for solver in stage_simulator.solvers:
        stats = solver.get_stats()
        for key, value in stats.items():
            total_stats[key] = (
                total_stats.get(key, 0) + value
            )

    worker_stats = engine.collect_worker_stats()
    for key, value in worker_stats.items():
        total_stats[key] = (
            total_stats.get(key, 0) + value
        )

    if WORKERS > 1:
        print(
            f"  (main process: {WORKERS} worker "
            f"processes, stats aggregated)",
        )

    for key, value in sorted(
        total_stats.items()
    ):
        print(f"  {key:>25}: {value:>10}")


def main() -> None:
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

    _print_mechanism_info(definition)
    _print_lever_overview(definition)

    # -------------------------------------------------
    # Target curve
    # -------------------------------------------------

    (
        target_input_angles_deg,
        target_output_angles_deg,
    ) = read_target_curve(TARGET_FILE)
    target_curve = TargetCurve.from_csv(
        TARGET_FILE,
    )

    target_input_angles_rad = tuple(
        math.radians(angle)
        for angle in target_input_angles_deg
    )
    target_output_angles_rad = tuple(
        math.radians(angle)
        for angle in target_output_angles_deg
    )

    _print_target_curve(
        target_input_angles_deg,
        target_output_angles_deg,
    )

    # -------------------------------------------------
    # Simulation setup
    # -------------------------------------------------

    motion = PointMotion(
        angles=target_input_angles_rad,
    )
    parameter_template = CsvParameterFactory.create(
        definition,
    )

    _print_simulation_setup(
        definition,
        target_input_angles_deg,
        parameter_template,
    )

    stage_simulator = StageSimulator()
    simulator = MechanismSimulator(
        motion=motion,
        stage_simulator=stage_simulator,
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
    # Initial population
    # -------------------------------------------------

    rng = random.Random(42)
    population_factory = PopulationFactory(
        random_generator=rng,
    )
    population = population_factory.create(
        parameter_template,
        size=POPULATION_SIZE,
    )

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

    engine = EvolutionEngine(
        population=population,
        evaluator=(
            optimizer.evaluate
            if WORKERS == 1
            else None
        ),
        selection_count=SELECTION_COUNT,
        reproduction=Reproduction(
            mutation=mutation_operator,
            random_generator=rng,
            crossover=Crossover(
                strategy="grouped",
                random_generator=rng,
            ),
        ),
        target_fitness=TARGET_FITNESS,
        max_generations=MAX_GENERATIONS,
        stagnation_limit=STAGNATION_LIMIT,
        stagnation_tolerance=(
            STAGNATION_TOLERANCE
        ),
        adaptive_strength=adaptive_controller,
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
        # Evolution
        # -------------------------------------------------

        print("\n" + "=" * 80)
        print("EVOLUTION ENGINE")
        print("=" * 80)
        print(f"\nPopulation size: {POPULATION_SIZE}")
        print(f"Selection count: {SELECTION_COUNT}")
        print(f"Children count:  {CHILDREN_COUNT}")
        print(f"Workers:         {WORKERS}" + (
            "  (sequential)"
            if WORKERS == 1
            else "  (parallel)"
        ))

        engine.evaluate_population()
        valid = sum(
            score < float("inf")
            for score in engine.scores.values()
        )

        print("\n" + "=" * 80)
        print("INITIAL POPULATION")
        print("=" * 80)
        print(
            f"Valid candidates: "
            f"{valid}/{len(engine.population)}",
        )
        if engine.best_score < float("inf"):
            print(
                f"Best initial fitness: "
                f"{engine.best_score:.8f}",
            )
        else:
            print(
                "Best initial fitness: "
                "inf (no valid candidates)",
            )

        if valid == 0:
            raise RuntimeError(
                "No valid candidates - "
                "mechanism is infeasible",
            )

        print("\n" + "=" * 80)
        print("EVOLUTION PROGRESS")
        print("=" * 80)
        print(
            "  Generation |         Fitness |  Improvement",
        )
        print("-" * 55)

        prev_best = float("inf")
        generation_count = 0

        for generation in engine.run(
            children_count=CHILDREN_COUNT,
        ):
            generation_count += 1
            improvement = (
                prev_best - engine.best_score
            )
            prev_best = engine.best_score
            if generation == 0:
                improvement_str = "          inf"
            else:
                improvement_str = (
                    f"{improvement:>12.8f}"
                )
            print(
                f"  {generation:3d} | "
                f"{engine.best_score:>16.8f} | "
                f"{improvement_str}",
            )

        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)

        print(f"\nStop reason: {engine.stop_reason}")
        print(
            f"Best fitness: "
            f"{engine.best_score:.12f}",
        )
        print(
            f"Generations run: {generation_count}",
        )

        print(
            "\nAdaptive mutation strength "
            "(1/5 success rule):",
        )
        print(
            f"  Initial strength : "
            f"{adaptive_controller.history[0]:.6f}",
        )
        print(
            f"  Final strength   : "
            f"{adaptive_controller.strength:.6f}",
        )
        print(
            f"  Adaptations      : "
            f"{len(adaptive_controller.history) - 1}",
        )
        if len(adaptive_controller.history) > 1:
            print(
                "  Strength history : "
                + ", ".join(
                    f"{s:.4f}"
                    for s in (
                        adaptive_controller.history
                    )
                ),
            )

        cache_stats = optimizer.get_cache_stats()
        print("\nCache statistics:")
        print(
            f"  Evaluations: "
            f"{cache_stats['evaluations']}",
        )
        print(
            f"  Cache hits: "
            f"{cache_stats['cache_hits']}",
        )
        print(
            f"  Cache misses: "
            f"{cache_stats['cache_misses']}",
        )
        print(
            f"  Cache size: "
            f"{cache_stats['cache_size']}",
        )
        if WORKERS > 1:
            print(
                "  (main process only - every "
                "worker keeps its own cache)",
            )

        best = engine.best_candidate
        if best is None:
            raise RuntimeError(
                "Optimization produced no candidate.",
            )

        # -------------------------------------------------
        # Best configuration, validation, simulation
        # -------------------------------------------------

        mechanism = _print_best_configuration(
            definition,
            builder,
            best,
        )

        simulation_results = (
            _print_validation_and_simulation(
                builder,
                simulator,
                mechanism,
                target_input_angles_deg,
                target_output_angles_deg,
            )
        )

        _print_solver_statistics(
            stage_simulator,
            engine,
        )

        # -------------------------------------------------
        # Optimized mechanism file
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
        print("\n" + "=" * 80)
        print("OPTIMIZED MECHANISM")
        print("=" * 80)
        print(
            f"\nOptimized mechanism written to: "
            f"{OPTIMIZED_FILE}",
        )
    finally:
        engine.close()

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------

    final_result = simulation_results[-1]
    drive_result = simulation_results[0]
    all_succeeded = all(
        result.success
        for result in simulation_results
    )
    if not all_succeeded:
        raise RuntimeError(
            "Simulation blocked - no complete "
            "result to plot.",
        )

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
        mechanism,
        simulation_results,
    ):
        plotter.add_mechanism_state(state)

    output_path = plotter.write(OUTPUT_FILE)

    print("=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    print(
        f"\nDrive angles : "
        f"{len(drive_result.input_angles)} points",
    )
    print(
        f"Output curve : "
        f"{len(final_result.output_angles)} points",
    )
    print(
        f"\nHTML document written to: {output_path}",
    )


if __name__ == "__main__":
    main()
