"""Tests for issue #128: weighted support points of the
target curves are not prioritized sufficiently.

The blocked mechanism was an unweighted endpoint
constraint: its fixed weight 1.0 outranked the braked
curve's per-point weights, pulling the result toward the
maximum input angle no matter how the weights were set.

The endpoint constraint weight now follows the weight
share of the braked curve's LAST support point (the very
point the constraint enforces), so per-point weights
govern all additive fitness terms.  Curves without
per-point weights keep the neutral weight 1.0.
"""

from __future__ import annotations

import math

import pytest

from analysis.target_curve import TargetCurve
from gui.mixed_brake_workflow import (
    _endpoint_constraint,
    mixed_brake_inputs_from_parts,
)
from mechanism_io.csv_reader import CsvReader

MIXER_CSV = """
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled,pivot_on
1,40,100,60,-40,40,0,0,60,0,0,0,1,,,lever3
2,20,80,50,-60,60,0,50,10,0,0,0,1,1,,
3,40,200,100,-30,30,0,180,0,0,0,0,1,2,,
4,20,80,50,-60,60,0,0,10,0,0,0,1,3,
""".strip()


@pytest.fixture
def mixer_definition(tmp_path):
    path = tmp_path / "mixer.csv"
    path.write_text(MIXER_CSV, encoding="utf-8")
    return CsvReader.read_mechanism(path)


def _weighted_curve(weights):
    return TargetCurve.from_points_strict(
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
        weights=tuple(weights),
    )


def _inputs(definition, braked):
    return mixed_brake_inputs_from_parts(
        definition,
        _weighted_curve((1.0, 1.0, 1.0)),
        braked,
    )


def test_endpoint_weight_follows_last_point_share(
    mixer_definition,
):
    # last support point carries weight 5, the others 1
    braked = _weighted_curve((1.0, 1.0, 5.0))
    inputs = _inputs(mixer_definition, braked)
    constraint = _endpoint_constraint(
        inputs,
        (0.0, math.radians(20)),
    )
    assert constraint.weight == pytest.approx(
        5.0 / 7.0,
    )


def test_endpoint_weight_neutral_without_weights(
    mixer_definition,
):
    braked = TargetCurve.from_points(
        input_angles=(
            math.radians(-10),
            math.radians(0),
            math.radians(10),
        ),
        output_angles=(
            math.radians(0),
            math.radians(10),
            math.radians(20),
        ),
    )
    inputs = _inputs(mixer_definition, braked)
    constraint = _endpoint_constraint(
        inputs,
        (0.0, math.radians(20)),
    )
    assert constraint.weight == 1.0


def test_endpoint_weight_lower_for_low_weight(
    mixer_definition,
):
    # issue #128: a LOW weight on the last support point
    # must weaken the endpoint term instead of the old
    # fixed 1.0 that ignored the weights completely
    low = _endpoint_constraint(
        _inputs(
            mixer_definition,
            _weighted_curve((10.0, 10.0, 1.0)),
        ),
        (0.0, math.radians(20)),
    )
    high = _endpoint_constraint(
        _inputs(
            mixer_definition,
            _weighted_curve((1.0, 1.0, 10.0)),
        ),
        (0.0, math.radians(20)),
    )
    assert low.weight < high.weight
    assert low.weight < 1.0
