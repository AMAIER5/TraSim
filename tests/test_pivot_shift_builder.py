"""tests/test_pivot_shift_builder.py

Tests for the pivot-shift builder (mechanics/pivot_shift_builder.py):
the brake state is the displacement of the common pivot of a
lever pair within a defined room; unbraked and braked builds
share IDENTICAL frozen coupling rods and differ only in the
displaced pivot.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)
from mechanics.pivot_shift_builder import (
    MechanismBuildError,
    PivotShiftBuilder,
)
from mechanism_io.csv_reader import CsvReader
from optimization.parameter_set import (
    ParameterSet,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)
MECHANISM_FILE = (
    EXAMPLES_DIR / "pivot_shift_mechanism.csv"
)


@pytest.fixture
def definition():
    return CsvReader.read_mechanism(MECHANISM_FILE)


@pytest.fixture
def builder(definition):
    return PivotShiftBuilder(
        definition,
        shift_lever_id=1,
    )


def motion():
    return MotionRange(
        start_angle=math.radians(-10.0),
        max_angle=math.radians(20.0),
        step=math.radians(2.0),
    )


def test_pair_pivots_resolved(definition, builder):
    assert builder.shifted_lever_ids == {1, 3}


def test_auto_resolves_unique_pair(definition):
    auto = PivotShiftBuilder(definition)
    assert auto.shifted_lever_ids == {1, 3}


def test_unknown_shift_lever_raises(definition):
    with pytest.raises(KeyError):
        PivotShiftBuilder(
            definition,
            shift_lever_id=99,
        )


def test_lever_without_partner_raises(definition):
    with pytest.raises(MechanismBuildError):
        PivotShiftBuilder(
            definition,
            shift_lever_id=2,
        )


def test_pivot_on_definition_rejected():
    mixed = CsvReader.read_mechanism(
        EXAMPLES_DIR / "mixed_brake_mechanism.csv",
    )
    with pytest.raises(MechanismBuildError):
        PivotShiftBuilder(mixed)


def test_unbraked_build_matches_standard_builder(
    definition,
    builder,
):
    standard = CsvMechanismBuilder(
        definition,
    ).build(ParameterSet(parameters=()))
    unbraked = builder.build((0.0, 0.0))
    assert len(unbraked.stages) == len(
        standard.stages,
    )
    for shifted, plain in zip(
        unbraked.stages,
        standard.stages,
    ):
        assert shifted.rod_length == (
            pytest.approx(plain.rod_length)
        )
        assert shifted.input_angle == (
            pytest.approx(plain.input_angle)
        )
        assert shifted.output_angle == (
            pytest.approx(plain.output_angle)
        )


def test_rod_lengths_frozen_between_positions(builder):
    rods = builder.rod_lengths()
    unbraked = builder.build((0.0, 0.0))
    braked = builder.build((2.0, -10.0))
    assert unbraked.stages[0].rod_length == (
        pytest.approx(rods[2])
    )
    assert braked.stages[0].rod_length == (
        pytest.approx(rods[2])
    )


def test_shift_moves_pair_pivot_only(builder):
    shift = (3.0, -4.0)
    mechanism = builder.build(shift)
    lever_pivots = {
        index: entry.lever.pivot
        for index, entry in enumerate(
            mechanism.levers,
            start=1,
        )
    }
    assert lever_pivots[1].x == pytest.approx(
        -60.0 + shift[0],
    )
    assert lever_pivots[1].y == pytest.approx(
        0.0 + shift[1],
    )
    assert lever_pivots[3].x == pytest.approx(
        -60.0 + shift[0],
    )
    assert lever_pivots[3].y == pytest.approx(
        0.0 + shift[1],
    )
    assert lever_pivots[2].x == pytest.approx(100.0)
    assert lever_pivots[2].y == pytest.approx(60.0)


def test_braked_curves_differ_from_unbraked(builder):
    simulator = MechanismSimulator(motion=motion())
    unbraked = simulator.simulate(
        builder.build((0.0, 0.0)),
    )[0]
    braked = simulator.simulate(
        builder.build((2.0, -8.0)),
    )[0]
    assert unbraked.success
    assert braked.success
    assert any(
        abs(a - b) > 1.0e-6
        for a, b in zip(
            unbraked.output_angles,
            braked.output_angles,
        )
    )


def test_unreachable_shift_raises_build_error(
    builder,
):
    with pytest.raises(MechanismBuildError):
        builder.build((5000.0, 5000.0))


def test_explicit_rod_length_locked():
    definition = CsvReader.read_mechanism(
        MECHANISM_FILE,
    )
    locked = PivotShiftBuilder(
        definition,
        shift_lever_id=1,
        rod_lengths={2: 150.0},
    )
    assert locked.rod_lengths() == {2: 150.0}
    with pytest.raises(MechanismBuildError):
        PivotShiftBuilder(
            definition,
            shift_lever_id=1,
            rod_lengths={99: 1.0},
        )
