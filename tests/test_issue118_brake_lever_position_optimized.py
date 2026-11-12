"""
Issue #118: during mixed-brake optimization the brake
lever position must be a real optimization degree of
freedom.  The evaluated brake positions follow from the
CANDIDATE's brake lever angle range (angle_start unbraked
up to angle_max fully braked), not from a fixed absolute
grid, so moving the brake lever's ``angle`` parameter moves
the brake lever instead of only compromising the geometry
between the target curves.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from analysis.brake_curve_fitness import (
    BrakeCurveFitness,
)
from analysis.target_curve import TargetCurve
from mechanics.mixed_brake_builder import (
    MixedBrakeBuilder,
)
from mechanism_io.csv_reader import CsvReader
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.mixed_brake_optimizer import (
    MixedBrakeOptimizer,
)
from optimization.parameter_set import ParameterSet
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange

MIXER_CSV = """
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled,pivot_on
1,40,100,60,-40,40,0,0,60,0,0,0,1,,,lever3
2,20,80,50,-60,60,0,50,10,0,0,0,1,1,,
3,40,200,100,-30,30,0,180,0,0,0,0,1,2,,
4,20,80,50,-60,60,0,0,10,0,0,0,1,3,,
""".strip()

BRAKE_POSITIONS_DEG = (0.0, 20.0)


@pytest.fixture
def mixer_definition(tmp_path):
    path = tmp_path / "mixer.csv"
    path.write_text(MIXER_CSV, encoding="utf-8")
    return CsvReader.read_mechanism(path)


def _optimizer(definition) -> MixedBrakeOptimizer:
    motion = MotionRange(
        start_angle=math.radians(-10),
        max_angle=math.radians(20),
        step=math.radians(2),
    )
    targets = tuple(
        TargetCurve.from_points(
            input_angles=(
                math.radians(-10),
                math.radians(0),
                math.radians(10),
            ),
            output_angles=(
                math.radians(-20 + 10 * index),
                math.radians(0),
                math.radians(20 - 10 * index),
            ),
        )
        for index in range(2)
    )
    return MixedBrakeOptimizer(
        builder=MixedBrakeBuilder(definition),
        simulator=MechanismSimulator(motion=motion),
        fitness=BrakeCurveFitness(),
        definition=definition,
        brake_positions=tuple(
            math.radians(angle)
            for angle in BRAKE_POSITIONS_DEG
        ),
        targets=targets,
    )


def _candidate_with_brake_angle(
    definition,
    angle_deg: float,
) -> ParameterSet:
    template = CsvParameterFactory.create(definition)
    parameters = tuple(
        replace(
            parameter,
            value=math.radians(angle_deg),
        )
        if parameter.name == "lever.3.angle"
        else parameter
        for parameter in template.parameters
    )
    return ParameterSet(parameters=parameters)


def test_brake_positions_follow_base_candidate(
    mixer_definition,
):
    """The base candidate (brake angle unchanged) keeps the
    base brake grid."""
    optimizer = _optimizer(mixer_definition)
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        0.0,
    )
    positions = optimizer.brake_positions_for(candidate)
    assert positions == pytest.approx(
        tuple(
            math.radians(angle)
            for angle in BRAKE_POSITIONS_DEG
        )
    )


def test_brake_positions_follow_candidate_angle(
    mixer_definition,
):
    """Moving the brake lever's angle parameter moves every
    evaluated brake position with it: the grid fractions of
    the base grid are re-applied to the candidate's own
    [angle_start, angle_max] range (brake lever 3:
    angle_max = 30 deg)."""
    optimizer = _optimizer(mixer_definition)
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        10.0,
    )
    positions = optimizer.brake_positions_for(candidate)
    expected = (
        math.radians(10.0),
        math.radians(10.0 + (20.0 / 30.0) * 20.0),
    )
    assert positions == pytest.approx(expected)


def test_brake_positions_clamped_to_lever_bounds(
    mixer_definition,
):
    """A brake angle at the upper bound collapses the grid
    onto angle_max and stays inside the admissible segment."""
    optimizer = _optimizer(mixer_definition)
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        30.0,
    )
    positions = optimizer.brake_positions_for(candidate)
    assert positions == pytest.approx(
        (
            math.radians(30.0),
            math.radians(30.0),
        )
    )


def test_evaluate_uses_candidate_brake_positions(
    mixer_definition,
):
    """The evaluation really builds and simulates at the
    candidate's brake positions: the cache keys contain
    exactly the positions returned by brake_positions_for."""
    optimizer = _optimizer(mixer_definition)
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        10.0,
    )
    fitness = optimizer.evaluate(candidate)
    assert math.isfinite(fitness)
    expected = optimizer.brake_positions_for(candidate)
    cached_betas = {
        beta for (_, beta) in optimizer._cache
    }
    assert cached_betas
    for beta in expected:
        assert beta in cached_betas


def test_brake_lever_angle_changes_simulation(
    mixer_definition,
):
    """Two candidates that differ only in the brake lever
    angle are simulated at different brake positions, so
    the brake lever position is not frozen during
    optimization (issue #118)."""
    optimizer = _optimizer(mixer_definition)
    unbraked_a = _candidate_with_brake_angle(
        mixer_definition,
        0.0,
    )
    shifted_b = _candidate_with_brake_angle(
        mixer_definition,
        10.0,
    )
    beta_a = optimizer.brake_positions_for(unbraked_a)[0]
    beta_b = optimizer.brake_positions_for(shifted_b)[0]
    assert beta_a != pytest.approx(beta_b)
    result_a = optimizer._cached_simulation(
        unbraked_a,
        beta_a,
    )
    result_b = optimizer._cached_simulation(
        shifted_b,
        beta_b,
    )
    assert result_a is not None
    assert result_b is not None
    assert result_a.output_angles != pytest.approx(
        result_b.output_angles
    )
