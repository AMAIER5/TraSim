"""tests/test_pivot_shift_optimizer.py

Tests for the pivot-shift optimizer pipeline
(optimization/pivot_shift_optimizer.py):

(1) mutation of all parameters inside the allowed ranges
    (shift parameters are part of the template),
(2) frozen coupling rod lengths in the unbraked position,
(3) unbraked transfer function,
(4) braked transfer function with IDENTICAL coupling rods,
(5) fitness from both transfer functions.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from analysis.brake_curve_fitness import (
    PENALTY_MAX_BLOCKING,
    BrakeCurveFitness,
)
from analysis.target_curve import TargetCurve
from mechanics.pivot_shift_builder import (
    PivotShiftBuilder,
)
from mechanism_io.csv_reader import CsvReader
from model.mechanism_definition import (
    MechanismDefinition,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.pivot_shift_optimizer import (
    PivotShiftOptimizer,
    shift_parameter_names,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)


def load_inputs():
    definition = CsvReader.read_mechanism(
        EXAMPLES_DIR / "pivot_shift_mechanism.csv",
    )
    targets = (
        TargetCurve.from_csv(
            EXAMPLES_DIR
            / "target_curve_unbraked.csv",
        ),
        TargetCurve.from_csv(
            EXAMPLES_DIR / "target_curve_braked.csv",
        ),
    )
    return definition, targets


def make_optimizer(
    definition,
    targets,
    *,
    builder=None,
    shift_effect_weight=1.0,
):
    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=math.radians(-10.0),
            max_angle=math.radians(20.0),
            step=math.radians(2.0),
        ),
    )
    return PivotShiftOptimizer(
        builder=(
            builder
            if builder is not None
            else PivotShiftBuilder(
                definition,
                shift_lever_id=1,
            )
        ),
        simulator=simulator,
        fitness=BrakeCurveFitness(
            weights=(1.0, 1.0),
            max_weight=1.0,
        ),
        definition=definition,
        targets=targets,
        shift_lever_id=1,
        shift_effect_weight=shift_effect_weight,
    )


def with_values(template, **overrides):
    parameters = [
        (
            replace(
                parameter,
                value=overrides[parameter.name],
            )
            if (
                parameter.name in overrides
            )
            else parameter
        )
        for parameter in template.parameters
    ]
    return ParameterSet(
        parameters=tuple(parameters),
    )


@pytest.fixture
def setup():
    definition, targets = load_inputs()
    template = (
        CsvParameterFactory.create_pivot_shift(
            definition,
            shift_lever_id=1,
        )
    )
    return definition, targets, template


def test_template_contains_shift_parameters(setup):
    _, _, template = setup
    names = {
        parameter.name
        for parameter in template.parameters
    }
    assert set(shift_parameter_names(1)) <= names
    # The shift room follows the shift lever's pivot limits
    # in the example CSV (lever 1: pivot (-60, 0),
    # x in [-75, -45], y in [-15, 10]):
    # shift.x in [-15, 15], shift.y in [-15, 10]
    # (an asymmetric room, identical per-axis limits).
    x = template.get(shift_parameter_names(1)[0])
    y = template.get(shift_parameter_names(1)[1])
    assert x.value == 0.0
    assert y.value == 0.0
    assert (x.minimum, x.maximum) == (-15.0, 15.0)
    assert (y.minimum, y.maximum) == (-15.0, 10.0)


def test_template_has_no_partner_pivot_parameters(
    setup,
):
    _, _, template = setup
    names = {
        parameter.name
        for parameter in template.parameters
    }
    # Lever 3 is the torsion-coupled partner of the shifted
    # lever 1 on the common pivot: its pivot follows lever 1,
    # so it owns no pivot parameters.
    for axis in ("x", "y", "z"):
        assert (
            f"lever.3.pivot.{axis}" not in names
        )
    assert "lever.3.length" in names


def test_shift_room_configurable(setup):
    definition, _, _ = setup
    template = (
        CsvParameterFactory.create_pivot_shift(
            definition,
            shift_lever_id=1,
            shift_bounds=(-5.0, 15.0),
        )
    )
    x = template.get("lever.1.shift.x")
    assert (x.minimum, x.maximum) == (-5.0, 15.0)


def test_pivot_parameters_follow_definition_limits(setup):
    """The pivot search-space parameters use the optional
    per-axis pivot limits of the mechanism definition
    instead of the symmetric default offset."""
    _, _, template = setup
    x = template.get("lever.2.pivot.x")
    y = template.get("lever.2.pivot.y")
    assert (x.minimum, x.maximum) == (85.0, 115.0)
    assert (y.minimum, y.maximum) == (45.0, 75.0)
    # Lever 3 defines no limits and is the pair partner
    # (no pivot parameters at all).
    names = {
        parameter.name
        for parameter in template.parameters
    }
    assert "lever.3.pivot.x" not in names


def test_shift_clamped_to_definition_limits(setup):
    """The braked pair pivot stays inside the shift lever's
    pivot limits even when the candidate's raw shift exceeds
    the room left around its optimized base pivot."""
    definition, targets, template = setup
    optimizer = make_optimizer(
        definition,
        targets,
    )
    # Base pivot moved to x = -50: the room left to the
    # x_max limit (-45) is a shift of +5.
    candidate = with_values(
        template,
        **{
            "lever.1.pivot.x": -50.0,
            "lever.1.shift.x": 15.0,
        },
    )
    assert optimizer.shift_for(
        candidate,
    ) == (5.0, 0.0)
    mechanism = optimizer._cached_mechanism(
        candidate,
        optimizer.shift_for(candidate),
    )
    assert mechanism is not None
    braked_pivot = mechanism.levers[0].lever.pivot
    assert braked_pivot.x == pytest.approx(
        -45.0,
    )



def test_base_candidate_scores_finite(setup):
    definition, targets, template = setup
    optimizer = make_optimizer(
        definition,
        targets,
    )
    fitness = optimizer.evaluate(template)
    assert math.isfinite(fitness)
    assert fitness < PENALTY_MAX_BLOCKING


def test_rods_identical_between_positions(setup):
    definition, targets, template = setup
    optimizer = make_optimizer(
        definition,
        targets,
    )
    shift = (2.0, -8.0)
    candidate = with_values(
        template,
        **dict(
            zip(
                shift_parameter_names(1),
                shift,
            )
        ),
    )
    unbraked = optimizer._cached_mechanism(
        candidate,
        (0.0, 0.0),
    )
    braked = optimizer._cached_mechanism(
        candidate,
        shift,
    )
    assert unbraked is not None
    assert braked is not None
    for plain, shifted in zip(
        unbraked.stages,
        braked.stages,
    ):
        assert plain.rod_length == (
            pytest.approx(shifted.rod_length)
        )
    assert any(
        plain.input_lever.pivot.x
        != pytest.approx(
            shifted.input_lever.pivot.x,
        )
        or plain.input_lever.pivot.y
        != pytest.approx(
            shifted.input_lever.pivot.y,
        )
        for plain, shifted in zip(
            unbraked.stages,
            braked.stages,
        )
    )


def test_zero_shift_reproduces_unbraked_curve(setup):
    definition, targets, template = setup
    optimizer = make_optimizer(
        definition,
        targets,
    )
    unbraked = optimizer._cached_simulation(
        template,
        (0.0, 0.0),
    )
    braked = optimizer._cached_simulation(
        template,
        (0.0, 0.0),
    )
    assert unbraked is not None
    assert braked is not None
    assert unbraked.output_angles == (
        pytest.approx(braked.output_angles)
    )


def test_pair_pivot_stays_common_after_parameter_overlay(
    setup,
):
    definition, targets, template = setup
    optimizer = make_optimizer(
        definition,
        targets,
    )
    candidate = with_values(
        template,
        **{
            "lever.1.pivot.x": -70.0,
            "lever.1.pivot.y": 5.0,
        },
    )
    applied = optimizer.apply_parameters(
        candidate,
    )
    leader = applied.get_lever(1)
    partner = applied.get_lever(3)
    assert partner.pivot.x == pytest.approx(
        leader.pivot.x,
    )
    assert partner.pivot.y == pytest.approx(
        leader.pivot.y,
    )


def test_unbuildable_unbraked_hard_penalty(setup):
    definition, targets, template = setup
    builder = PivotShiftBuilder(
        definition,
        shift_lever_id=1,
        rod_lengths={2: 1.0e6},
    )
    optimizer = make_optimizer(
        definition,
        targets,
        builder=builder,
    )
    assert (
        optimizer.evaluate(template)
        == PENALTY_MAX_BLOCKING
    )


def test_unbuildable_braked_adds_skip_penalty(setup):
    definition, targets, template = setup
    optimizer = make_optimizer(
        definition,
        targets,
    )
    # A shift outside the room bounds cannot be expressed
    # by the template, so an explicit unreachable candidate
    # is constructed here.
    from optimization.parameter import Parameter

    shift_x_name, shift_y_name = (
        shift_parameter_names(1)
    )
    parameters = [
        parameter
        for parameter in template.parameters
        if parameter.name
        not in (shift_x_name, shift_y_name)
    ]
    parameters.extend([
        Parameter(
            name=shift_x_name,
            minimum=-5000.0,
            maximum=5000.0,
            value=5000.0,
        ),
        Parameter(
            name=shift_y_name,
            minimum=-5000.0,
            maximum=5000.0,
            value=5000.0,
        ),
    ])
    unreachable = ParameterSet(
        parameters=tuple(parameters),
    )
    unbraked_mechanism = PivotShiftBuilder(
        definition,
        shift_lever_id=1,
    ).build((0.0, 0.0))
    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=math.radians(-10.0),
            max_angle=math.radians(20.0),
            step=math.radians(2.0),
        ),
    )
    fitness = BrakeCurveFitness(
        weights=(1.0, 1.0),
        max_weight=1.0,
    )
    result = simulator.simulate(
        unbraked_mechanism,
    )[0]
    expected = fitness.evaluate(
        [(result, targets[0])],
        weights=(1.0,),
    )
    got = optimizer.evaluate(unreachable)
    skip_penalty = got - expected
    assert skip_penalty > 0.0
    # The skip penalty is proportional (factor 10) to the
    # mean squared difference between the braked and the
    # unbraked target curves.
    support = tuple(result.input_angles)
    msd = sum(
        (
            targets[1].evaluate(angle)
            - targets[0].evaluate(angle)
        )
        ** 2
        for angle in support
    ) / len(support)
    assert skip_penalty == pytest.approx(
        10.0 * msd,
        rel=1e-9,
    )


def test_collapsed_shift_pays_shift_effect(setup):
    definition, targets, template = setup
    strict = make_optimizer(
        definition,
        targets,
        shift_effect_weight=1.0,
    )
    lax = make_optimizer(
        definition,
        targets,
        shift_effect_weight=0.0,
    )
    collapsed = with_values(
        template,
        **dict(
            zip(
                shift_parameter_names(1),
                (0.0, 0.0),
            )
        ),
    )
    assert strict.evaluate(
        collapsed,
    ) > lax.evaluate(collapsed)


def test_example_optimization_reaches_finite_score(
    setup,
):
    definition, targets, template = setup
    import random

    from optimization.evolution_engine import (
        EvolutionEngine,
    )
    from optimization.parameter_mutation import (
        ParameterMutation,
    )
    from optimization.population_factory import (
        PopulationFactory,
    )
    from optimization.reproduction import (
        Reproduction,
    )

    optimizer = make_optimizer(
        definition,
        targets,
    )
    rng = random.Random(42)
    population = PopulationFactory(
        random_generator=rng,
        initial_spread=0.1,
    ).create(template, size=10)
    engine = EvolutionEngine(
        population=population,
        evaluator=optimizer.evaluate,
        selection_count=4,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                strength=0.05,
                random_generator=rng,
            ),
        ),
        max_generations=3,
    )
    engine.evaluate_population()
    for _ in enumerate(
        engine.run(children_count=10),
    ):
        pass
    assert engine.best_score < float("inf")


def test_definition_pivot_limits_roundtrip(setup, tmp_path):
    """The optional pivot limits are part of the mechanism
    definition: read from and written to the CSV."""
    from mechanism_io.csv_writer import CsvWriter

    definition, _, _ = setup
    lever1 = definition.get_lever(1)
    assert lever1.pivot_x_min == -75.0
    assert lever1.pivot_x_max == -45.0
    assert lever1.pivot_y_min == -15.0
    assert lever1.pivot_y_max == 10.0
    assert lever1.pivot_z_min is None
    partner = definition.get_lever(3)
    assert not partner.has_pivot_limits

    path = tmp_path / "roundtrip.csv"
    CsvWriter.write_mechanism(definition, path)
    reloaded = CsvReader.read_mechanism(path)
    for original in definition.levers:
        loaded = reloaded.get_lever(original.id)
        assert (loaded.pivot_x_min, loaded.pivot_x_max) == (
            original.pivot_x_min,
            original.pivot_x_max,
        )
        assert (loaded.pivot_y_min, loaded.pivot_y_max) == (
            original.pivot_y_min,
            original.pivot_y_max,
        )
        assert (loaded.pivot_z_min, loaded.pivot_z_max) == (
            original.pivot_z_min,
            original.pivot_z_max,
        )


def test_inconsistent_definition_limits_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,pivot_x_min,pivot_x_max
1,40,100,60,-12,12,0,-60,0,0,0,0,1,-40,-75
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        CsvReader.read_mechanism(path)


def test_pivot_outside_definition_limits_rejected(tmp_path):
    path = tmp_path / "outside.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,pivot_x_min
1,40,100,60,-12,12,0,-60,0,0,0,0,1,-50
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        CsvReader.read_mechanism(path)


def _wide_definition(definition):
    """Definition copy whose drive lever may rotate far
    beyond the shaft support, so a lever angle parameter of
    +-90 deg stays inside the parameter range."""
    from dataclasses import replace as _replace

    return MechanismDefinition(
        tuple(
            (
                _replace(
                    lever,
                    angle_min=math.radians(-360.0),
                    angle_max=math.radians(360.0),
                )
                if lever.id == 1
                else lever
            )
            for lever in definition.levers
        ),
    )


def test_drive_shaft_offset_follows_candidate_angle(
    setup,
):
    """The target input angle describes the rotation of a
    VIRTUAL DRIVE SHAFT; the optimizer forms the initial
    angular offset ``drive angle - mean(input values)``."""
    definition, targets, _ = setup
    optimizer = make_optimizer(
        _wide_definition(definition),
        targets,
    )
    from optimization.csv_parameter_factory import (
        CsvParameterFactory,
    )

    template = CsvParameterFactory.create_pivot_shift(
        _wide_definition(definition),
        shift_lever_id=1,
    )
    assert optimizer.drive_shaft_offset(
        template,
    ) == pytest.approx(0.0)
    moved = with_values(
        template,
        **{
            "lever.1.angle": math.radians(90.0),
        },
    )
    offset = optimizer.drive_shaft_offset(
        moved,
    )
    mean_shaft = optimizer._support_mean
    assert offset == pytest.approx(
        math.radians(90.0) - mean_shaft,
    )
    anchored = optimizer.anchored_support(moved)
    assert sum(anchored) / len(
        anchored,
    ) == pytest.approx(
        math.radians(90.0),
        rel=1e-6,
    )


def test_simulated_input_angles_stay_in_shaft_frame(
    setup,
):
    """Results are reported in SHAFT coordinates even
    though the drive lever sweeps its own lever-angle
    frame (the initial angular offset between the virtual
    drive shaft and the start lever)."""
    definition, targets, _ = setup
    optimizer = make_optimizer(
        _wide_definition(definition),
        targets,
    )
    from optimization.csv_parameter_factory import (
        CsvParameterFactory,
    )

    template = CsvParameterFactory.create_pivot_shift(
        _wide_definition(definition),
        shift_lever_id=1,
    )
    moved = with_values(
        template,
        **{
            "lever.1.angle": math.radians(10.0),
        },
    )
    result = optimizer._cached_simulation(
        moved,
        (0.0, 0.0),
    )
    assert result is not None
    expected = [
        math.radians(value)
        for value in (
            -10,
            -8,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
        )
    ]
    assert len(result.input_angles) == len(
        expected,
    )
    for got, want in zip(
        result.input_angles,
        expected,
    ):
        assert got == pytest.approx(
            want,
            abs=1e-9,
        )


def test_shift_effect_survives_single_point_curves(setup):
    definition, targets, _ = setup
    optimizer = make_optimizer(
        definition,
        targets,
    )
    from simulation.simulation_result import (
        SimulationResult,
    )

    angle = 0.5
    one_point = SimulationResult(
        input_angles=(angle,),
        output_angles=(0.5,),
        success=False,
        blocked_at=angle,
    )
    deviation = optimizer._pair_shift_effect_deviation(
        (
            (one_point, targets[0]),
            (one_point, targets[1]),
        ),
    )
    expected = (
        (
            one_point.output_angles[0]
            - one_point.output_angles[0]
        )
        - (
            targets[1].evaluate(angle)
            - targets[0].evaluate(angle)
        )
    ) ** 2
    assert deviation == pytest.approx(expected)
