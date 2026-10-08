"""examples/optimize_mixed_brake.py

End-to-end evolutionary optimization of a mixer mechanism
whose output lever pivot sits ON a brake lever (pivot_on).

Setup
-----
- mechanism CSV with three levers:
  * Lever 1: input lever (tight +/-12 deg angle bounds; the
    stage motion validator checks the COMPLETE input range)
  * Lever 2: output lever (generous +/-120 deg angle bounds)
  * Lever 3: brake lever carrying the pivot of lever 2 via
    pivot_on=lever3@180.  It is oriented so that it displaces
    the pivot TRANSVERSALLY (angle ~180 deg), avoiding the
    kinematic branch jump a radial displacement causes.
- Two target curves (unbraked / braked) derived from the real
  reachable behaviour of the mechanism, sharing the common
  endpoint characteristic (output returns to 0 deg at the
  maximum input angle).

Optimization
------------
MixedBrakeOptimizer evaluates every candidate over both brake
positions (brake lever at 180 deg and 190 deg).  The
EvolutionEngine is consumed as an ITERATOR (engine.run(...)),
because OptimizerRunner.run() alone does not update
best_score.

User I/O angles are in DEGREES, internal calculations use
RADIANS.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from analysis.brake_curve_fitness import BrakeCurveFitness
from analysis.endpoint_constraint import EndpointConstraint
from analysis.target_curve import TargetCurve
from mechanism_io.csv_reader import CsvReader
from mechanics.mixed_brake_builder import MixedBrakeBuilder
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.evolution_engine import EvolutionEngine
from optimization.mixed_brake_optimizer import (
    MixedBrakeOptimizer,
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
from simulation.motion_range import MotionRange

BASE_DIR = Path(__file__).parent
MECHANISM_FILE = (
    BASE_DIR / "mixed_brake_mechanism.csv"
)
TARGET_UNBRAKED_FILE = (
    BASE_DIR / "target_curve_unbraked.csv"
)
TARGET_BRAKED_FILE = (
    BASE_DIR / "target_curve_braked.csv"
)

BRAKE_POSITIONS_DEG = (180.0, 190.0)

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
    builder = MixedBrakeBuilder(definition)

    print("=" * 70)
    print("MIXED BRAKE OPTIMIZATION")
    print("=" * 70)
    print(f"\nMechanism: {len(definition.levers)} levers")
    for lever in definition.levers:
        pivot_on = (
            f", pivot_on={lever.pivot_on}"
            if lever.pivot_on is not None
            else ""
        )
        print(
            f"  Lever {lever.id}: pivot="
            f"({lever.pivot.x:.1f}, "
            f"{lever.pivot.y:.1f}, "
            f"{lever.pivot.z:.1f}), "
            f"length={lever.length_start:.1f}, "
            f"angle=[{math.degrees(lever.angle_min):.1f}"
            f"\u00b0, "
            f"{math.degrees(lever.angle_max):.1f}\u00b0]"
            f"{pivot_on}"
        )

    targets = (
        TargetCurve.from_csv(TARGET_UNBRAKED_FILE),
        TargetCurve.from_csv(TARGET_BRAKED_FILE),
    )
    brake_positions = tuple(
        math.radians(beta)
        for beta in BRAKE_POSITIONS_DEG
    )

    print(
        f"\nBrake positions (deg): "
        f"{BRAKE_POSITIONS_DEG}"
    )
    print(f"Target unbraked: {TARGET_UNBRAKED_FILE.name}")
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

    optimizer = MixedBrakeOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
        definition=definition,
        brake_positions=brake_positions,
        targets=targets,
    )

    template = CsvParameterFactory.create(definition)
    print(f"\nParameters: {len(template.parameters)}")
    for param in template.parameters:
        if "angle" in param.name:
            print(
                f"  {param.name}: "
                f"[{math.degrees(param.minimum):.1f}"
                f"\u00b0, "
                f"{math.degrees(param.maximum):.1f}"
                f"\u00b0], "
                f"default="
                f"{math.degrees(param.value):.1f}"
                f"\u00b0"
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
    print(
        f"Best fitness: {engine.best_score:.8f}"
    )
    print(f"Generations run: {generations}")

    stats = optimizer.get_cache_stats()
    print(f"\nCache statistics:")
    for key, value in stats.items():
        print(f"  {key:>15}: {value}")

    if engine.best_candidate is None:
        raise RuntimeError("No best candidate found")

    print("\n" + "=" * 70)
    print("BEST MECHANISM GEOMETRY")
    print("=" * 70)
    best = engine.best_candidate.values()
    for lever in definition.levers:
        length = best.get(
            f"lever.{lever.id}.length",
            lever.length_start,
        )
        angle = best.get(
            f"lever.{lever.id}.angle",
            lever.angle_start,
        )
        print(
            f"  Lever {lever.id}: "
            f"length={length:.2f} mm, "
            f"angle={math.degrees(angle):.2f}\u00b0"
        )

    print("\nSIMULATED CURVES OF BEST CANDIDATE")
    for beta_deg, target in zip(
        BRAKE_POSITIONS_DEG,
        targets,
    ):
        result = optimizer._cached_simulation(
            engine.best_candidate,
            math.radians(beta_deg),
        )
        if result is None:
            print(
                f"\n  brake={beta_deg:.1f}\u00b0: "
                "build error"
            )
            continue
        print(
            f"\n  brake={beta_deg:.1f}\u00b0 "
            f"(success={result.success}):"
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
