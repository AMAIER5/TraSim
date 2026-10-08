"""
tests/test_brake_curve_fitness.py

Tests for BrakeCurveFitness and EndpointConstraint.

Covers:
- Two curves both matched well -> small fitness
- One curve matched poorly -> max term dominates
- Blocking in one brake position -> hard penalty
- Endpoint deviation -> quadratic constraint term
"""
from __future__ import annotations

from math import isclose

import pytest

from analysis.brake_curve_fitness import (
    PENALTY_MAX_BLOCKING,
    BrakeCurveFitness,
)
from analysis.endpoint_constraint import EndpointConstraint
from analysis.target_curve import TargetCurve
from simulation.simulation_result import SimulationResult


def identity_target() -> TargetCurve:
    return TargetCurve.from_points(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 2.0),
    )


def doubled_target() -> TargetCurve:
    return TargetCurve.from_points(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 2.0, 4.0),
    )


def exact_simulation(target: TargetCurve) -> SimulationResult:
    inputs = (0.0, 1.0, 2.0)
    return SimulationResult(
        input_angles=inputs,
        output_angles=tuple(
            target.function(angle) for angle in inputs
        ),
        success=True,
    )


def offset_simulation(
    target: TargetCurve,
    offset: float,
) -> SimulationResult:
    inputs = (0.0, 1.0, 2.0)
    return SimulationResult(
        input_angles=inputs,
        output_angles=tuple(
            target.function(angle) + offset for angle in inputs
        ),
        success=True,
    )


def test_both_curves_good_small_fitness():
    pairs = [
        (exact_simulation(identity_target()), identity_target()),
        (exact_simulation(doubled_target()), doubled_target()),
    ]
    fitness = BrakeCurveFitness().evaluate(pairs)
    assert isclose(fitness, 0.0, abs_tol=1e-9)


def test_one_curve_bad_max_term_dominates():
    good = (exact_simulation(identity_target()), identity_target())
    bad_offset = 1.0
    bad = (
        offset_simulation(doubled_target(), bad_offset),
        doubled_target(),
    )

    only_good = BrakeCurveFitness().evaluate([good])
    good_and_bad = BrakeCurveFitness().evaluate([good, bad])

    expected = 2.0 * bad_offset
    assert isclose(good_and_bad, only_good + expected)
    assert good_and_bad > bad_offset


def test_weights_scale_errors():
    good = (exact_simulation(identity_target()), identity_target())
    bad_offset = 2.0
    bad = (
        offset_simulation(doubled_target(), bad_offset),
        doubled_target(),
    )
    unweighted = BrakeCurveFitness().evaluate([good, bad])
    weighted = BrakeCurveFitness(
        weights=(1.0, 0.5)
    ).evaluate([good, bad])
    assert isclose(weighted, unweighted - 1.0)


def test_blocking_in_one_position_penalty():
    good = (exact_simulation(identity_target()), identity_target())
    blocked = SimulationResult(
        input_angles=(0.0, 1.0),
        output_angles=(0.0, 1.0),
        success=False,
        blocked_at=0.5,
    )
    pairs = [good, (blocked, identity_target())]
    fitness = BrakeCurveFitness().evaluate(pairs)
    assert fitness >= PENALTY_MAX_BLOCKING
    assert isclose(fitness, PENALTY_MAX_BLOCKING)


def test_empty_pairs_raise():
    with pytest.raises(ValueError):
        BrakeCurveFitness().evaluate([])


def test_weight_count_mismatch_raises():
    pair = (exact_simulation(identity_target()), identity_target())
    with pytest.raises(ValueError):
        BrakeCurveFitness(weights=(1.0, 1.0)).evaluate([pair])


def test_endpoint_constraint_quadratic():
    constraint = EndpointConstraint(end_angle=2.0)
    assert isclose(constraint.evaluate((0.0, 1.0, 2.0)), 0.0)
    assert isclose(constraint.evaluate((0.0, 1.0, 3.0)), 1.0)
    assert isclose(constraint.evaluate((0.0, 1.0, 4.0)), 4.0)


def test_endpoint_constraint_grows_quadratically_in_fitness():
    target = identity_target()
    exact = exact_simulation(target)
    dev1 = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 3.0),
        success=True,
    )
    dev2 = SimulationResult(
        input_angles=(0.0, 1.0, 2.0),
        output_angles=(0.0, 1.0, 4.0),
        success=True,
    )
    constraint = EndpointConstraint(end_angle=2.0)
    base = BrakeCurveFitness()
    with_c = BrakeCurveFitness(endpoint_constraint=constraint)

    pairs_exact = [(exact, target)]
    pairs_dev1 = [(dev1, target)]
    pairs_dev2 = [(dev2, target)]

    assert isclose(
        with_c.evaluate(pairs_dev1) - base.evaluate(pairs_dev1),
        1.0,
    )
    assert isclose(
        with_c.evaluate(pairs_dev2) - base.evaluate(pairs_dev2),
        4.0,
    )
    assert (
        base.evaluate(pairs_dev1) - base.evaluate(pairs_exact)
        < with_c.evaluate(pairs_dev1) - with_c.evaluate(pairs_exact)
    )


def test_penalty_max_blocking_exported():
    assert PENALTY_MAX_BLOCKING == 100000.0
