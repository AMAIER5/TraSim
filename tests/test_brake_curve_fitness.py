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


def discrete_offset_simulation(
    target: TargetCurve,
    offset: float,
) -> SimulationResult:
    inputs = target.input_angles
    return SimulationResult(
        input_angles=inputs,
        output_angles=tuple(
            target.evaluate(angle) + offset
            for angle in inputs
        ),
        success=True,
    )


def test_point_weights_scale_curves_globally():
    """Issue #105: the point weights must act across curves.

    The unbraked curve is weighted 2 at every point, the braked
    curve 1 at every point; both are off by the same offset, so
    the unbraked curve must contribute twice as much to the
    weighted sum as the braked one.
    """
    unbraked = TargetCurve.from_points_strict(
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        weights=(2.0, 2.0, 2.0),
    )
    braked = TargetCurve.from_points_strict(
        (0.0, 1.0, 2.0),
        (0.0, 2.0, 4.0),
        weights=(1.0, 1.0, 1.0),
    )
    offset = 1.0
    pairs = [
        (discrete_offset_simulation(unbraked, offset), unbraked),
        (discrete_offset_simulation(braked, offset), braked),
    ]
    fitness = BrakeCurveFitness().evaluate(pairs)
    # both pair errors are 1.0 (weighted mean within each curve);
    # curve scales are 2.0 and 1.0:
    # weighted_sum = 1*2*1 + 1*1*1 = 3, max_term = 1*max(1, 1) = 1
    assert isclose(fitness, 4.0)


def test_point_weights_change_relative_curve_influence():
    """Issue #105: a higher-weighted curve must influence the
    fitness more strongly when its error is the larger one."""
    unbraked = TargetCurve.from_points_strict(
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        weights=(2.0, 2.0, 2.0),
    )
    braked = TargetCurve.from_points_strict(
        (0.0, 1.0, 2.0),
        (0.0, 2.0, 4.0),
        weights=(1.0, 1.0, 1.0),
    )
    pairs_weighted = [
        (discrete_offset_simulation(unbraked, 2.0), unbraked),
        (discrete_offset_simulation(braked, 0.0), braked),
    ]
    pairs_unweighted = [
        (
            discrete_offset_simulation(
                TargetCurve.from_points_strict(
                    (0.0, 1.0, 2.0),
                    (0.0, 1.0, 2.0),
                ),
                2.0,
            ),
            TargetCurve.from_points_strict(
                (0.0, 1.0, 2.0),
                (0.0, 1.0, 2.0),
            ),
        ),
        (discrete_offset_simulation(braked, 0.0), braked),
    ]
    weighted = BrakeCurveFitness().evaluate(pairs_weighted)
    unweighted = BrakeCurveFitness().evaluate(pairs_unweighted)
    # weighted:   1*2*2 + 1*1*0 + 1*2 = 6
    # unweighted: 1*1*2 + 1*1*0 + 1*2 = 4
    assert isclose(weighted, 6.0)
    assert isclose(unweighted, 4.0)
    assert weighted > unweighted


def test_default_point_weights_keep_neutral_scale():
    """Curves with the default weight 1 at every point are
    unaffected by the cross-curve scaling (issue #105)."""
    first = TargetCurve.from_points_strict(
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
    )
    second = TargetCurve.from_points_strict(
        (0.0, 1.0, 2.0),
        (0.0, 2.0, 4.0),
    )
    offset = 1.0
    pairs = [
        (discrete_offset_simulation(first, offset), first),
        (discrete_offset_simulation(second, offset), second),
    ]
    fitness = BrakeCurveFitness().evaluate(pairs)
    # weighted_sum = 1*1 + 1*1 = 2, max_term = 1
    assert isclose(fitness, 3.0)
