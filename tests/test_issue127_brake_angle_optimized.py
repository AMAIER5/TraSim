"""Tests for issue #127: the brake lever's fully braked
position always ended up at the maximum allowed value
(the brake lever's ``angle_max`` bound), even when that
did not improve the mechanics.

The fully braked position is now an optimizable
parameter (``lever.<id>.brake_angle``) bounded by the
brake lever's admissible angle segment
``[angle_min, angle_max]``:

- the mixed brake parameter template exposes it,
- the optimizer applies it as the candidate brake
  lever's ``angle_max`` (never below its unbraked
  start angle, so the travel stays >= 0),
- the brake grid of such a candidate follows the
  candidate's own travel,
- the optimized definition keeps the optimized braked
  position so reports and CSV match the candidate.

Candidates WITHOUT the parameter (e.g. plain
``CsvParameterFactory`` templates used in unit tests)
keep the rigid-translation behaviour of issue #118.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from analysis.brake_curve_fitness import (
    BrakeCurveFitness,
)
from analysis.target_curve import TargetCurve
from gui.mixed_brake_workflow import (
    _parameter_definition,
    mixed_brake_inputs_from_parts,
    mixed_brake_parameter_template,
)
from mechanics.mixed_brake_builder import (
    MixedBrakeBuilder,
)
from mechanism_io.csv_reader import CsvReader
from model.mechanism_definition import MechanismDefinition
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.mixed_brake_optimizer import (
    MixedBrakeOptimizer,
    brake_angle_parameter_name,
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

BRAKE_ID = 3
BRAKE_POSITIONS_DEG = (0.0, 20.0)


@pytest.fixture
def mixer_definition(tmp_path):
    path = tmp_path / "mixer.csv"
    path.write_text(MIXER_CSV, encoding="utf-8")
    return CsvReader.read_mechanism(path)


def _simple_inputs(definition):
    return mixed_brake_inputs_from_parts(
        definition,
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
                math.radians(0),
                math.radians(10),
                math.radians(20),
            ),
        ),
    )


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
                math.radians(-20 + 20 * index),
                math.radians(0),
                math.radians(20 - 20 * index),
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
    brake_angle,
) -> ParameterSet:
    brake = definition.get_lever(BRAKE_ID)
    template = mixed_brake_parameter_template(
        _simple_inputs(definition),
    )
    return ParameterSet(
        parameters=tuple(
            replace(
                parameter,
                value=brake_angle,
            )
            if parameter.name
            == brake_angle_parameter_name(
                brake.id,
            )
            else parameter
            for parameter in template.parameters
        ),
    )


def test_template_exposes_brake_angle_parameter(
    mixer_definition,
):
    inputs = _simple_inputs(mixer_definition)
    brake = mixer_definition.get_lever(BRAKE_ID)
    template = mixed_brake_parameter_template(
        inputs,
    )
    parameter = next(
        parameter
        for parameter in template.parameters
        if parameter.name
        == brake_angle_parameter_name(
            brake.id,
        )
    )
    assert parameter.minimum == pytest.approx(
        brake.angle_min,
    )
    assert parameter.maximum == pytest.approx(
        brake.angle_max,
    )
    assert parameter.value == pytest.approx(
        brake.angle_max,
    )
    # the standard CSV parameters are all still present
    for name in (
        f"lever.{brake.id}.length",
        f"lever.{brake.id}.angle",
    ):
        assert name in template.values()


def test_brake_grid_follows_candidate_brake_angle(
    mixer_definition,
):
    optimizer = _optimizer(mixer_definition)
    brake = mixer_definition.get_lever(BRAKE_ID)
    base_start = brake.angle_start
    base_span = brake.angle_max - base_start

    plain = CsvParameterFactory.create(
        mixer_definition,
    )
    base = optimizer.brake_positions_for(plain)
    # a candidate WITHOUT the parameter keeps the base
    # rigid translation (issue #118 behaviour): the grid
    # fractions of the base brake positions (0 and 2/3
    # here) are re-applied to the base span unchanged
    assert base[0] == pytest.approx(base_start)
    assert base[-1] == pytest.approx(
        base_start
        + (2.0 / 3.0) * base_span,
    )

    # a candidate WITH the parameter places the fully
    # braked position inside the admissible segment
    braked_deg = math.radians(15.0)
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        braked_deg,
    )
    positions = optimizer.brake_positions_for(
        candidate,
    )
    assert positions[0] == pytest.approx(base_start)
    assert positions[-1] == pytest.approx(
        braked_deg,
    )


def test_apply_parameters_sets_candidate_brake_position(
    mixer_definition,
):
    optimizer = _optimizer(mixer_definition)
    braked_deg = math.radians(15.0)
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        braked_deg,
    )
    applied = optimizer._apply_parameters(candidate)
    brake = applied.get_lever(BRAKE_ID)
    # issue #127: the candidate's braked position is held in
    # the dedicated brake_position attribute; angle_max stays a
    # pure optimization bound.
    assert brake.brake_position == pytest.approx(
        braked_deg,
    )
    assert brake.resolved_brake_position == (
        pytest.approx(braked_deg)
    )
    base_brake = mixer_definition.get_lever(BRAKE_ID)
    assert brake.angle_max == pytest.approx(
        base_brake.angle_max,
    )


def test_braked_end_never_below_unbraked_start(
    mixer_definition,
):
    optimizer = _optimizer(mixer_definition)
    brake = mixer_definition.get_lever(BRAKE_ID)
    below_start = (
        brake.angle_start - math.radians(5.0)
    )
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        below_start,
    )
    applied = optimizer._apply_parameters(candidate)
    applied_brake = applied.get_lever(BRAKE_ID)
    assert (
        applied_brake.resolved_brake_position
        >= applied_brake.angle_start
    )


def test_parameter_definition_keeps_optimized_brake_end(
    mixer_definition,
):
    braked_deg = math.radians(15.0)
    candidate = _candidate_with_brake_angle(
        mixer_definition,
        braked_deg,
    )
    definition = _parameter_definition(
        _simple_inputs(mixer_definition),
        candidate,
    )
    brake = definition.get_lever(BRAKE_ID)
    assert brake.resolved_brake_position == (
        pytest.approx(braked_deg)
    )
    base_brake = mixer_definition.get_lever(BRAKE_ID)
    assert brake.angle_max == pytest.approx(
        base_brake.angle_max,
    )


def test_template_honors_dedicated_brake_position(
    mixer_definition,
):
    """Issue #127: the separate initial braked position
    attribute (``brake_position``) becomes the start value
    of the optimizable ``brake_angle`` parameter, while
    ``angle_max`` stays a pure optimization bound."""
    from dataclasses import replace as _replace

    braked_deg = math.radians(15.0)
    definition = MechanismDefinition(
        tuple(
            _replace(
                lever,
                brake_position=(braked_deg
                                if lever.id == BRAKE_ID
                                else None),
            )
            for lever in mixer_definition.levers
        ),
    )
    inputs = mixed_brake_inputs_from_parts(
        definition,
        _curve(),
        _curve(),
    )
    template = mixed_brake_parameter_template(
        inputs,
    )
    parameter = next(
        parameter
        for parameter in template.parameters
        if parameter.name
        == brake_angle_parameter_name(
            BRAKE_ID,
        )
    )
    assert parameter.value == pytest.approx(
        braked_deg,
    )
    assert parameter.maximum == pytest.approx(
        definition.get_lever(BRAKE_ID).angle_max,
    )
    assert inputs.brake_max_deg == pytest.approx(
        math.degrees(braked_deg),
    )


def _curve() -> TargetCurve:
    return TargetCurve.from_points(
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
    )


def test_brake_grid_follows_dedicated_brake_position(
    mixer_definition,
):
    """Issue #127: the brake grid is spanned between the
    unbraked start and the DEDICATED braked position, so
    editing it (independently of angle_max) directly moves
    the evaluated brake positions."""
    from dataclasses import replace as _replace

    braked_deg = 15.0
    definition = MechanismDefinition(
        tuple(
            _replace(
                lever,
                brake_position=(math.radians(
                    braked_deg,
                )
                    if lever.id == BRAKE_ID
                    else None),
            )
            for lever in mixer_definition.levers
        ),
    )
    optimizer = _optimizer(definition)
    plain = mixed_brake_parameter_template(
        mixed_brake_inputs_from_parts(
            definition,
            _curve(),
            _curve(),
        ),
    )
    positions = optimizer.brake_positions_for(plain)
    brake = definition.get_lever(BRAKE_ID)
    assert positions[0] == pytest.approx(
        brake.angle_start,
    )
    assert positions[-1] == pytest.approx(
        math.radians(braked_deg),
    )
    # the bound angle_max is untouched
    base_brake = mixer_definition.get_lever(BRAKE_ID)
    assert brake.angle_max == pytest.approx(
        base_brake.angle_max,
    )
