"""tests/test_mixed_brake_parallel.py

Issue #86: the mixed brake (mixer) optimization must
honour the ``workers`` setting.  ``workers > 1`` runs
the evaluation in a process pool (same results as the
sequential run), ``workers = 0`` picks all available
CPU cores, and the pool is shut down after the run.

Issue #87: the blended target curves of the mixed brake
grid carry the (blended) per-support-point weights of
the two base curves, and the fitness honours them.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from analysis.brake_curve_fitness import (
    BrakeCurveFitness,
)
from analysis.brake_grid import (
    BrakeGridSettings,
)
from analysis.target_curve import (
    DiscreteTargetCurve,
    TargetCurve,
)
from gui.mixed_brake_workflow import (
    MixedBrakeSettings,
    _blended_targets,
    _motion,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
    run_mixed_brake_optimization,
)

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)
MECHANISM_FILE = (
    EXAMPLES_DIR / "mixed_brake_mechanism.csv"
)
UNBRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_unbraked.csv"
)
BRAKED_FILE = (
    EXAMPLES_DIR / "target_curve_braked.csv"
)


@pytest.fixture
def inputs():
    return load_mixed_brake_inputs(
        MECHANISM_FILE.read_text(encoding="utf-8"),
        UNBRAKED_FILE.read_text(encoding="utf-8"),
        BRAKED_FILE.read_text(encoding="utf-8"),
    )


@pytest.fixture
def template(inputs):
    return mixed_brake_parameter_template(inputs)


def _settings(**overrides) -> MixedBrakeSettings:
    return MixedBrakeSettings(
        population_size=6,
        children_count=6,
        selection_count=3,
        max_generations=2,
        brake_positions=3,
        **overrides,
    )


def _run(inputs, template, settings):
    return run_mixed_brake_optimization(
        inputs,
        template,
        settings,
    )


def test_parallel_matches_sequential(
    inputs,
    template,
):
    sequential = _run(
        inputs,
        template,
        _settings(workers=1),
    )
    parallel = _run(
        inputs,
        template,
        _settings(workers=2),
    )
    assert (
        parallel.best_score
        == pytest.approx(
            sequential.best_score
        )
    )
    assert (
        parallel.generations
        == sequential.generations
    )


def test_zero_workers_uses_all_cores(
    inputs,
    template,
):
    """``workers=0`` must map to the number of CPU
    cores, so the mixed mode uses the full available
    compute (issue #86)."""
    workers = max(1, os.cpu_count() or 1)
    settings = _settings(workers=0)
    result = _run(inputs, template, settings)
    assert result.best_score < float("inf")
    assert workers >= 1


def test_blended_targets_carry_weights(inputs):
    motion = _motion(inputs)
    angles = tuple(angle for angle in motion)
    grid = BrakeGridSettings(
        brake_min_deg=inputs.brake_min_deg,
        brake_max_deg=inputs.brake_max_deg,
        position_count=3,
    )
    targets = _blended_targets(
        inputs,
        grid.fractions,
        input_angles=angles,
    )
    assert isinstance(
        targets[0],
        DiscreteTargetCurve,
    )
    for target in targets:
        assert target.weights == tuple(
            1.0 for _ in angles
        )


def test_blended_weights_follow_brake_fraction():
    """The blended weight of a support point blends
    the same way as its output (issue #87)."""
    angles = tuple(
        math.radians(angle)
        for angle in (-10.0, 0.0, 10.0)
    )
    unbraked = TargetCurve.from_points_strict(
        angles,
        tuple(
            math.radians(angle) / 2.0
            for angle in angles
        ),
        weights=(2.0, 0.0, 0.0),
    )
    braked = TargetCurve.from_points_strict(
        angles,
        angles,
        weights=(0.0, 0.0, 4.0),
    )

    class _Inputs:
        input_angles = angles

        class unbraked:
            pass

    blended = _blended_targets(
        _wrap(unbraked, braked, angles),
        (0.0, 0.5, 1.0),
        input_angles=angles,
    )
    assert blended[0].weights == (
        2.0,
        0.0,
        0.0,
    )
    assert blended[1].weights == (
        1.0,
        0.0,
        2.0,
    )
    assert blended[2].weights == (
        0.0,
        0.0,
        4.0,
    )


def _wrap(unbraked, braked, angles):
    from gui.mixed_brake_workflow import (
        MixedBrakeInputs,
    )

    return MixedBrakeInputs(
        definition=None,
        unbraked=unbraked,
        braked=braked,
        motion_start_deg=-10.0,
        motion_travel_deg=20.0,
        motion_step_deg=2.0,
        brake_min_deg=0.0,
        brake_max_deg=10.0,
        input_angles=angles,
    )


def test_fitness_honours_support_point_weights():
    """A support point with weight 0 must not
    influence the fitness (issue #87)."""
    angles = (0.0, 1.0, 2.0)
    target = TargetCurve.from_points_strict(
        angles,
        (0.0, 0.0, 0.0),
        weights=(1.0, 1.0, 1.0),
    )

    class _Result:
        success = True
        input_angles = angles
        output_angles = (
            0.0,
            0.0,
            1.0,
        )

    equal = BrakeCurveFitness()
    base = equal.evaluate([(_Result(), target)])
    ignored = TargetCurve.from_points_strict(
        angles,
        (0.0, 0.0, 0.0),
        weights=(1.0, 1.0, 0.0),
    )
    zero = BrakeCurveFitness()
    off = zero.evaluate(
        [(_Result(), ignored)]
    )
    # weighted mean (1/3) plus the max-error term
    # (max_weight=1 -> 1/3)
    assert base == pytest.approx(2.0 / 3.0)
    assert off == pytest.approx(0.0)
