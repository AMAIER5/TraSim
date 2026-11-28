"""examples/optimize_pivot_shift.py

End-to-end evolutionary optimization of a mechanism whose
brake state is the DISPLACEMENT of the common pivot of a
lever pair within a defined room (pivot-shift approach,
replacement of the modelled brake lever).

Setup
-----
- mechanism CSV with three levers, all with FIXED pivots:
  * Lever 1: input lever (tight +/-12 deg angle bounds); it
    shares its pivot with its torsion-coupled partner
    Lever 3 (the lever pair of the shifted pivot)
  * Lever 2: output lever (generous +/-120 deg angle bounds),
    driven by Lever 1 through a coupling rod
  * Lever 3: partner of the pair on the common pivot
- The candidate pipeline:
  (1) mutation of all parameters (length, angle, pivot and
      the pair pivot's shift dx/dy/dz within the room; the
      room follows the shift lever's optional pivot_x_min..
      pivot_z_max columns of the mechanism CSV),
  (2) coupling rod lengths frozen in the UNSHIFTED (unbraked)
      position,
  (3) transfer function of the unbraked position,
  (4) transfer function of the braked position with
      IDENTICAL coupling rods - both states differ only in
      the displaced pivot,
  (5) fitness from both transfer functions.
- Two target curves (unbraked / braked) shared with the
  mixed-brake example.

User I/O angles are in DEGREES, internal calculations use
RADIANS.  Lengths are in millimetres.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from analysis.brake_curve_fitness import (
    BrakeCurveFitness,
)
from analysis.endpoint_constraint import (
    EndpointConstraint,
)
from analysis.target_curve import TargetCurve
from analysis.transmission_angle import (
    MIN_TRANSMISSION_ANGLE,
    TransmissionAngleConstraint,
)
from mechanics.pivot_shift_builder import (
    PivotShiftBuilder,
)
from mechanism_io.csv_reader import CsvReader
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.parameter_mutation import (
    ParameterMutation,
)
from optimization.pivot_shift_optimizer import (
    PivotShiftOptimizer,
)
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import Reproduction
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange

BASE_DIR = Path(__file__).parent
MECHANISM_FILE = (
    BASE_DIR / "pivot_shift_mechanism.csv"
)
TARGET_UNBRAKED_FILE = (
    BASE_DIR / "target_curve_unbraked.csv"
)
TARGET_BRAKED_FILE = (
    BASE_DIR / "target_curve_braked.csv"
)

#: Id of one lever of the displaced lever pair.
SHIFT_LEVER_ID = 1

#: The room of the displacement follows the shift lever's
#: OPTIONAL per-axis pivot limits in the mechanism CSV
#: (columns pivot_x_min .. pivot_z_max): the braked pair
#: pivot stays inside the same admissible room in which the
#: unbraked pivot itself may be optimized.

MIN_TRANSMISSION_ANGLE_DEG = math.degrees(
    MIN_TRANSMISSION_ANGLE
)
TRANSMISSION_WEIGHT = 1.0

MOTION_START_DEG = -10.0
MOTION_TRAVEL_DEG = 20.0
MOTION_STEP_DEG = 2.0

POPULATION_SIZE = 40
CHILDREN_COUNT = 40
SELECTION_COUNT = 12
MAX_GENERATIONS = 40
MUTATION_STRENGTH = 0.03
RANDOM_SEED = 42


def main() -> None:
    definition = CsvReader.read_mechanism(
        MECHANISM_FILE
    )
    builder = PivotShiftBuilder(
        definition,
        shift_lever_id=SHIFT_LEVER_ID,
    )

    print("=" * 70)
    print("PIVOT SHIFT OPTIMIZATION")
    print("=" * 70)
    print(f"\nMechanism: {len(definition.levers)} levers")
    for lever in definition.levers:
        marker = (
            "  (shifted pair pivot)"
            if lever.id in builder.shifted_lever_ids
            else ""
        )
        print(
            f"  Lever {lever.id}: pivot="
            f"({lever.pivot.x:.1f}, "
            f"{lever.pivot.y:.1f}, "
            f"{lever.pivot.z:.1f}), "
            f"length={lever.length_start:.1f}, "
            f"angle=["
            f"{math.degrees(lever.angle_min):.1f}"
            "\u00b0, "
            f"{math.degrees(lever.angle_max):.1f}\u00b0"
            f"]{marker}"
        )
    print(
        f"Shifted lever pair: "
        f"{sorted(builder.shifted_lever_ids)}"
    )

    unbraked = TargetCurve.from_csv(
        TARGET_UNBRAKED_FILE,
    )
    braked = TargetCurve.from_csv(
        TARGET_BRAKED_FILE,
    )
    print(
        f"\nTarget unbraked: {TARGET_UNBRAKED_FILE.name}"
    )
    print(f"Target braked:   {TARGET_BRAKED_FILE.name}")

    motion = MotionRange(
        start_angle=math.radians(MOTION_START_DEG),
        max_angle=math.radians(MOTION_TRAVEL_DEG),
        step=math.radians(MOTION_STEP_DEG),
    )
    simulator = MechanismSimulator(motion=motion)
    fitness = BrakeCurveFitness(
        weights=(1.0, 1.0),
        max_weight=1.0,
        endpoint_constraint=EndpointConstraint(
            end_angle=0.0,
            weight=1.0,
        ),
    )
    transmission_constraint = (
        TransmissionAngleConstraint(
            minimum_angle=math.radians(
                MIN_TRANSMISSION_ANGLE_DEG
            ),
            weight=TRANSMISSION_WEIGHT,
        )
    )
    optimizer = PivotShiftOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
        definition=definition,
        targets=(unbraked, braked),
        shift_lever_id=SHIFT_LEVER_ID,
        transmission_angle_constraint=(
            transmission_constraint
        ),
    )
    template = CsvParameterFactory.create_pivot_shift(
        definition,
        shift_lever_id=SHIFT_LEVER_ID,
    )
    print(f"\nParameters: {len(template.parameters)}")
    for param in template.parameters:
        if "angle" in param.name:
            print(
                f"  {param.name}: "
                f"[{math.degrees(param.minimum):.1f}"
                "\u00b0, "
                f"{math.degrees(param.maximum):.1f}"
                "\u00b0], "
                f"default="
                f"{math.degrees(param.value):.1f}"
                "\u00b0"
            )
        else:
            print(
                f"  {param.name}: "
                f"[{param.minimum:.1f}, "
                f"{param.maximum:.1f}], "
                f"default={param.value:.1f}"
            )

    rng = random.Random(RANDOM_SEED)
    population_factory = PopulationFactory(
        random_generator=rng,
        initial_spread=0.1,
    )
    population = population_factory.create(
        template,
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
        target_fitness=0.005,
        max_generations=MAX_GENERATIONS,
        stagnation_limit=20,
        stagnation_tolerance=1e-8,
    )

    engine.evaluate_population()
    valid = sum(
        score < float("inf")
        for score in engine.scores.values()
    )
    print("\n" + "=" * 70)
    print("INITIAL POPULATION")
    print("=" * 70)
    print(f"Valid candidates: {valid}/{len(engine.population)}")
    print(
        "Best initial fitness: "
        f"{engine.best_score:.8f}"
        if engine.best_score < float("inf")
        else "Best initial fitness: inf"
    )
    if valid == 0:
        raise RuntimeError(
            "No valid candidates - mechanism setup is "
            "infeasible"
        )

    print("\n" + "=" * 70)
    print("EVOLUTION PROGRESS")
    print("=" * 70)
    print(
        f"{'Generation':>10} | {'Best fitness':>16}"
    )
    print("-" * 40)
    generations = 0
    for _ in engine.run(children_count=CHILDREN_COUNT):
        generations += 1
        print(
            f"{generations:>10d} | "
            f"{engine.best_score:>16.8f}"
        )

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Stop reason: {engine.stop_reason}")
    print(f"Best fitness: {engine.best_score:.8f}")
    print(f"Generations run: {generations}")

    stats = optimizer.get_cache_stats()
    print("\nCache statistics:")
    for key, value in stats.items():
        print(f"  {key:>15}: {value}")

    if engine.best_candidate is None:
        raise RuntimeError("No best candidate found")

    best = engine.best_candidate.values()
    print("\n" + "=" * 70)
    print("BEST MECHANISM GEOMETRY")
    print("=" * 70)
    optimized_definition = optimizer.apply_parameters(
        engine.best_candidate,
    )
    for lever in optimized_definition.levers:
        marker = (
            "  (shifted pair pivot)"
            if lever.id in builder.shifted_lever_ids
            else ""
        )
        print(
            f"  Lever {lever.id}: "
            f"length={lever.length_start:.2f} mm, "
            f"angle={math.degrees(lever.angle_start):.2f}\u00b0, "
            f"pivot=({lever.pivot.x:.2f}, "
            f"{lever.pivot.y:.2f}){marker}"
        )
    dx = best[f"lever.{SHIFT_LEVER_ID}.shift.x"]
    dy = best[f"lever.{SHIFT_LEVER_ID}.shift.y"]
    dz = best.get(
        f"lever.{SHIFT_LEVER_ID}.shift.z",
        0.0,
    )
    print(
        f"  Braked pair pivot shift: "
        f"({dx:.2f}, {dy:.2f}, {dz:.2f}) mm"
    )

    print("\nSIMULATED CURVES OF BEST CANDIDATE")
    for label, shift in (
        ("unbraked", (0.0, 0.0, 0.0)),
        ("braked", (dx, dy, dz)),
    ):
        result = optimizer._cached_simulation(
            engine.best_candidate,
            shift,
        )
        if result is None:
            print(
                f"\n  {label} (shift={shift}): build error"
            )
            continue
        print(
            f"\n  {label} (shift=({shift[0]:.2f}, "
            f"{shift[1]:.2f}), "
            f"success={result.success}):"
        )
        for angle, output in zip(
            result.input_angles,
            result.output_angles,
        ):
            print(
                f"    {math.degrees(angle):7.2f}\u00b0"
                f" -> {math.degrees(output):8.3f}\u00b0"
            )


if __name__ == "__main__":
    main()
