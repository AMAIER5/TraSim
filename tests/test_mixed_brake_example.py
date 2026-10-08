"""tests/test_mixed_brake_example.py

Tests for the runnable mixed-brake example
(examples/optimize_mixed_brake.py):

- example CSVs load and produce a valid mixer topology
- both brake positions build and simulate successfully
- target curves share the endpoint characteristic
- a short optimization run reaches a finite best score
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from analysis.brake_curve_fitness import (
    BrakeCurveFitness,
)
from analysis.endpoint_constraint import (
    EndpointConstraint,
)
from analysis.target_curve import TargetCurve
from mechanism_io.csv_reader import CsvReader
from mechanics.mixed_brake_builder import (
    MixedBrakeBuilder,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.evolution_engine import (
    EvolutionEngine,
)
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

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)
MECHANISM_FILE = (
    EXAMPLES_DIR / "mixed_brake_mechanism.csv"
)
TARGET_UNBRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_unbraked.csv"
)
TARGET_BRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_braked.csv"
)

BRAKE_POSITIONS_DEG = (180.0, 190.0)


@pytest.fixture
def example_definition():
    return CsvReader.read_mechanism(
        MECHANISM_FILE
    )


@pytest.fixture
def example_targets():
    return (
        TargetCurve.from_csv(
            TARGET_UNBRAKED_FILE
        ),
        TargetCurve.from_csv(
            TARGET_BRAKED_FILE
        ),
    )


def test_example_files_exist():
    for path in (
        MECHANISM_FILE,
        TARGET_UNBRAKED_FILE,
        TARGET_BRAKED_FILE,
        EXAMPLES_DIR
        / "optimize_mixed_brake.py",
    ):
        assert path.is_file(), f"missing {path}"


def test_mixer_topology(example_definition):
    mixers = [
        lever
        for lever in example_definition.levers
        if lever.pivot_on is not None
    ]
    assert len(mixers) == 1
    mixer = mixers[0]
    assert (
        mixer.pivot_reference.lever_id == 3
    )
    assert (
        mixer.pivot_reference.angle_deg
        == pytest.approx(180.0)
    )
    brake = example_definition.get_lever(3)
    assert math.degrees(brake.angle_min) == (
        pytest.approx(170.0)
    )
    assert math.degrees(brake.angle_max) == (
        pytest.approx(190.0)
    )
    input_lever = example_definition.get_lever(1)
    assert math.degrees(
        input_lever.angle_max
    ) == pytest.approx(12.0)


def test_brake_positions_within_bounds(
    example_definition,
):
    brake = example_definition.get_lever(3)
    for beta_deg in BRAKE_POSITIONS_DEG:
        assert brake.angle_min <= math.radians(
            beta_deg
        ) <= brake.angle_max


def test_targets_share_endpoint_characteristic(
    example_targets,
):
    unbraked, braked = example_targets
    max_input = math.radians(10.0)
    inputs = (0.0, max_input)
    end_unbraked = unbraked.sample(
        inputs
    ).output_angles[1]
    end_braked = braked.sample(
        inputs
    ).output_angles[1]
    assert end_braked == (
        pytest.approx(0.0, abs=1e-6)
    )
    assert abs(end_unbraked) > abs(
        end_braked
    )
    assert end_unbraked > 0.0


def _simulator() -> MechanismSimulator:
    motion = MotionRange(
        start_angle=math.radians(-10.0),
        max_angle=math.radians(20.0),
        step=math.radians(2.0),
    )
    return MechanismSimulator(motion=motion)


def test_both_brake_positions_build_and_simulate(
    example_definition,
):
    builder = MixedBrakeBuilder(
        example_definition
    )
    simulator = _simulator()
    for beta_deg in BRAKE_POSITIONS_DEG:
        mechanism = builder.build(
            math.radians(beta_deg)
        )
        results = simulator.simulate(mechanism)
        assert len(results) > 0
        result = results[-1]
        assert result.success
        assert len(result.input_angles) == 11


def _optimizer(definition, targets):
    return MixedBrakeOptimizer(
        builder=MixedBrakeBuilder(definition),
        simulator=_simulator(),
        fitness=BrakeCurveFitness(
            weights=(1.0, 1.0),
            max_weight=1.0,
            endpoint_constraint=(
                EndpointConstraint(
                    end_angle=0.0,
                    weight=1.0,
                )
            ),
        ),
        definition=definition,
        brake_positions=tuple(
            math.radians(beta)
            for beta in BRAKE_POSITIONS_DEG
        ),
        targets=targets,
    )


def test_short_optimization_finite_fitness(
    example_definition,
    example_targets,
):
    import random

    optimizer = _optimizer(
        example_definition,
        example_targets,
    )
    template = CsvParameterFactory.create(
        example_definition
    )
    rng = random.Random(42)
    population_factory = PopulationFactory(
        random_generator=rng,
        initial_spread=0.1,
    )
    population = population_factory.create(
        template,
        size=8,
    )
    engine = EvolutionEngine(
        population=population,
        evaluator=optimizer.evaluate,
        selection_count=4,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                strength=0.03,
                random_generator=rng,
            ),
        ),
        max_generations=3,
        stagnation_limit=10,
        stagnation_tolerance=1e-8,
    )
    engine.evaluate_population()
    valid = sum(
        score < float("inf")
        for score in engine.scores.values()
    )
    assert valid > 0
    for _ in engine.run(children_count=8):
        pass
    assert (
        engine.best_candidate is not None
    )
    assert engine.best_score < 0.5
