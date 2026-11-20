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
    for name in shift_parameter_names(1):
        parameter = template.get(name)
        assert parameter.value == 0.0
        assert parameter.minimum == -20.0
        assert parameter.maximum == 20.0


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
