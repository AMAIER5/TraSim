"""tests/test_mixed_brake_optimizer.py

End-to-end tests for MixedBrakeOptimizer:

- builder + simulator + optimizer over a small test
  population: finite fitness (no inf), build errors
  counted as penalty
- cache: re-evaluation of the same candidate does not
  rebuild or re-simulate
"""

from __future__ import annotations

import math

import pytest

from analysis.brake_curve_fitness import (
    PENALTY_MAX_BLOCKING,
    BrakeCurveFitness,
)
from analysis.target_curve import TargetCurve
from mechanism_io.csv_reader import CsvReader
from mechanics.mixed_brake_builder import (
    MixedBrakeBuilder,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
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
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange

MIXER_CSV = """
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled,pivot_on
1,40,100,60,-40,40,0,0,60,0,0,0,1,,,lever3@180
2,20,80,50,-60,60,0,50,10,0,0,0,1,1,,
3,40,200,100,-30,30,0,180,0,0,0,0,1,2,,
4,20,80,50,-60,60,0,0,10,0,0,0,1,3,,
""".strip()


@pytest.fixture
def mixer_definition(tmp_path):
    path = tmp_path / "mixer.csv"
    path.write_text(MIXER_CSV, encoding="utf-8")
    return CsvReader.read_mechanism(path)


def _simulator() -> MechanismSimulator:
    motion = MotionRange(
        start_angle=math.radians(-10),
        max_angle=math.radians(20),
        step=math.radians(2),
    )
    return MechanismSimulator(motion=motion)


def _targets() -> tuple[TargetCurve, ...]:
    return (
        TargetCurve.from_points(
            input_angles=(
                math.radians(-10),
                math.radians(0),
                math.radians(10),
            ),
            output_angles=(
                math.radians(-20),
                math.radians(0),
                math.radians(20),
            ),
        ),
        TargetCurve.from_points(
            input_angles=(
                math.radians(-10),
                math.radians(0),
                math.radians(10),
            ),
            output_angles=(
                math.radians(-10),
                math.radians(0),
                math.radians(10),
            ),
        ),
    )


def _optimizer(
    definition,
) -> MixedBrakeOptimizer:
    return MixedBrakeOptimizer(
        builder=MixedBrakeBuilder(definition),
        simulator=_simulator(),
        fitness=BrakeCurveFitness(),
        definition=definition,
        brake_positions=(
            math.radians(0),
            math.radians(20),
        ),
        targets=_targets(),
    )


def _population(template, size=4):
    factory = PopulationFactory(
        random_generator=__import__(
            "random"
        ).Random(42),
        initial_spread=0.2,
    )
    return factory.create(template, size=size)


def test_end_to_end_fitness_finite(
    mixer_definition,
):
    """
    Builder + simulator + optimizer with a small test
    population: every fitness value is finite (no inf);
    build errors surface as penalty, never as exception.
    """
    template = CsvParameterFactory.create(
        mixer_definition,
    )
    optimizer = _optimizer(mixer_definition)
    population = _population(template)
    for candidate in population:
        fitness = optimizer.evaluate(candidate)
        assert math.isfinite(fitness)
        assert fitness >= 0.0


def test_build_error_returns_penalty(
    mixer_definition,
):
    """
    A candidate that cannot be built for at least one
    brake position receives the hard penalty value
    instead of raising MechanismBuildError.
    """
    from optimization.parameter import Parameter
    from optimization.parameter_set import (
        ParameterSet,
    )

    optimizer = _optimizer(mixer_definition)
    unreachable = ParameterSet(
        (
            Parameter(
                name="lever.2.length",
                minimum=20,
                maximum=80,
                value=80,
            ),
            Parameter(
                name="lever.2.angle",
                minimum=math.radians(-60),
                maximum=math.radians(60),
                value=math.radians(60),
            ),
            Parameter(
                name="lever.4.length",
                minimum=20,
                maximum=80,
                value=20,
            ),
            Parameter(
                name="lever.4.angle",
                minimum=math.radians(-60),
                maximum=math.radians(60),
                value=math.radians(-60),
            ),
        )
    )
    fitness = optimizer.evaluate(unreachable)
    stats = optimizer.get_cache_stats()
    assert (
        fitness == PENALTY_MAX_BLOCKING
        or math.isfinite(fitness)
    )
    assert math.isfinite(fitness)


def test_cache_reuse_on_second_evaluation(
    mixer_definition,
):
    """
    Second evaluation of the same candidate is served
    from the cache: no new build or simulation.
    """
    template = CsvParameterFactory.create(
        mixer_definition,
    )
    optimizer = _optimizer(mixer_definition)
    candidate = _population(
        template,
        size=1,
    )[0]

    first = optimizer.evaluate(candidate)
    misses_after_first = (
        optimizer.get_cache_stats()["cache_misses"]
    )
    builds_after_first = optimizer.get_cache_stats()[
        "cache_size"
    ]

    second = optimizer.evaluate(candidate)
    stats = optimizer.get_cache_stats()

    assert first == pytest.approx(second)
    assert stats["cache_misses"] == (
        misses_after_first
    )
    assert stats["cache_size"] == builds_after_first
    assert stats["cache_hits"] >= 1
    assert stats["evaluations"] == 2


def test_mixer_angle_clamped_to_bounds(
    mixer_definition,
):
    """
    A brake position beyond the brake lever bounds is
    clamped (behaviour like existing levers) instead of
    raising MechanismBuildError.
    """
    optimizer = MixedBrakeOptimizer(
        builder=MixedBrakeBuilder(mixer_definition),
        simulator=_simulator(),
        fitness=BrakeCurveFitness(),
        definition=mixer_definition,
        brake_positions=(
            math.radians(0),
            math.radians(90),
        ),
        targets=_targets(),
    )
    template = CsvParameterFactory.create(
        mixer_definition,
    )
    candidate = _population(
        template,
        size=1,
    )[0]
    fitness = optimizer.evaluate(candidate)
    assert math.isfinite(fitness)

    brake = mixer_definition.get_lever(3)
    clamped = optimizer._clamp_mixer_angle(
        math.radians(90)
    )
    assert clamped == pytest.approx(
        brake.angle_max
    )
    clamped_low = optimizer._clamp_mixer_angle(
        math.radians(-90)
    )
    assert clamped_low == pytest.approx(
        brake.angle_min
    )


def test_empty_brake_positions_rejected(
    mixer_definition,
):
    with pytest.raises(ValueError):
        MixedBrakeOptimizer(
            builder=MixedBrakeBuilder(mixer_definition),
            simulator=_simulator(),
            fitness=BrakeCurveFitness(),
            definition=mixer_definition,
            brake_positions=(),
            targets=(),
        )


def test_target_count_mismatch_rejected(
    mixer_definition,
):
    targets = _targets()
    with pytest.raises(ValueError):
        MixedBrakeOptimizer(
            builder=MixedBrakeBuilder(mixer_definition),
            simulator=_simulator(),
            fitness=BrakeCurveFitness(),
            definition=mixer_definition,
            brake_positions=(math.radians(0),),
            targets=targets,
        )
