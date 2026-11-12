"""\
tests/test_issue115_non_blocking_unbraked.py

Regression test for issue #115: "non-blocking solution for
unbreaked curve must never lead to blocking solution of
breaked curve".

When a non-blocking solution for the unbraked curve is
possible, no solution may be generated that blocks in the
braked position.  Blocking solutions must always have a worse
fitness value than non-blocking solutions; in the worst case
the brake lever travel between the unbraked and the braked
position is reduced to 0 deg, because that definitely does
not generate a blocking solution.

With the issue's mechanism every brake position beta != 0 is
geometrically unreachable (issue #111/#113), so the best
candidate is non-blocking only at the unbraked 0 % position.
The generated solution must therefore carry a zero brake
travel (brake lever ``angle_max`` clamped onto ``angle_start``),
so reloading it can never produce a blocking braked position.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from analysis.brake_curve_fitness import (
    PENALTY_MAX_BLOCKING,
    BrakeCurveFitness,
)
from analysis.brake_grid import (
    BrakeGridSettings,
    make_brake_grid,
)
from analysis.transmission_angle import (
    TransmissionAngleConstraint,
)
from gui.mixed_brake_workflow import (
    MixedBrakeSettings,
    _blended_targets,
    _braked_blocks_but_unbraked_works,
    _brake_lever_id,
    _endpoint_constraint,
    _motion,
    _reduce_brake_travel_to_zero,
    _reference_output_angle,
    build_mixed_brake_grid,
    load_mixed_brake_inputs,
    mixed_brake_inputs_from_parts,
    mixed_brake_parameter_template,
    run_mixed_brake_optimization,
)
from mechanics.mixed_brake_builder import (
    MixedBrakeBuilder,
)
from optimization.mixed_brake_optimizer import (
    MixedBrakeOptimizer,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.simulation_result import SimulationResult

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)

MECHANISM_CSV = """\
id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on
1;50;100;73,05799607;60;300;133,651956;0;0;239;1;0;0;;;;;;
2;50;100;76,9929424;100;260;153,1432978;0;0;-261;1;0;0;;;;1;;7
3;50;100;91,73216337;-90;90;11,8383644;0;0;-261;1;0;0;;;;;2;7
4;50;100;62,09694189;-70;70;-5,795825305;0;-40;-1020;1;0;0;;;;3;;
5;50;100;92,07312009;60;300;147,4731758;-815;-40;-1020;1;0;0;;;;;4;
6;70;85;82;30;130;80;-850;-218;-4020;0;1;0;;;;5;;
7;2000;2050;2000;-10;10;0;0;-2000;-261;1;0;0;;;;;;
"""

UNBRAKED_CSV = """\
input_angle;output_angle
-35;-25
-30;-23
-25;-20
-20;-17
-10;-9
0;0
10;12
20;24
25;31
30;38
35;46
"""

BRAKED_CSV = """\
input_angle;output_angle
-35;19
-30;27
-25;34
-20;40
-10;50
0;60
10;68
20;76
25;79
30;82
35;85
"""


def _blocking_inputs():
    return load_mixed_brake_inputs(
        MECHANISM_CSV,
        UNBRAKED_CSV,
        BRAKED_CSV,
    )


def _example_inputs():
    return load_mixed_brake_inputs(
        (EXAMPLES_DIR / "mixed_brake_mechanism.csv")
        .read_text(encoding="utf-8"),
        (EXAMPLES_DIR / "target_curve_unbraked.csv")
        .read_text(encoding="utf-8"),
        (EXAMPLES_DIR / "target_curve_braked.csv")
        .read_text(encoding="utf-8"),
    )


def _optimizer(inputs):
    grid = BrakeGridSettings(
        brake_min_deg=inputs.brake_min_deg,
        brake_max_deg=inputs.brake_max_deg,
        position_count=5,
    )
    brake_positions = make_brake_grid(grid)
    motion = _motion(inputs)
    input_angles = tuple(angle for angle in motion)
    targets = _blended_targets(
        inputs,
        grid.fractions,
        input_angles=input_angles,
    )
    return MixedBrakeOptimizer(
        builder=MixedBrakeBuilder(inputs.definition),
        simulator=MechanismSimulator(motion=motion),
        fitness=BrakeCurveFitness(
            weights=tuple(1.0 for _ in brake_positions),
            max_weight=1.0,
            endpoint_constraint=_endpoint_constraint(
                inputs,
                brake_positions,
            ),
        ),
        definition=inputs.definition,
        brake_positions=brake_positions,
        targets=targets,
        transmission_angle_constraint=(
            TransmissionAngleConstraint(
                minimum_angle=math.radians(20.0),
                weight=1.0,
            )
        ),
        anchor_output=True,
        reference_output_angle=(
            _reference_output_angle(inputs)
        ),
    )


def _small_settings() -> MixedBrakeSettings:
    return MixedBrakeSettings(
        population_size=8,
        children_count=8,
        selection_count=4,
        max_generations=2,
        brake_positions=5,
        workers=1,
    )


def test_unbraked_non_blocking_braked_unbuildable():
    """Documents the issue's situation: the mechanism is
    non-blocking at the unbraked 0 % position but cannot be
    built at any braked position."""
    inputs = _blocking_inputs()
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
        position_count=5,
    )
    assert grid.states[0] is not None
    assert grid.states[-1] is None
    optimizer = _optimizer(inputs)
    fitness = optimizer.evaluate(
        mixed_brake_parameter_template(inputs)
    )
    assert fitness < 1.0
    assert fitness < PENALTY_MAX_BLOCKING


def test_blocking_solution_scores_worse_than_non_blocking():
    """Blocking solutions must always have a worse fitness
    value than non-blocking solutions (issue #115)."""
    inputs = _blocking_inputs()
    optimizer = _optimizer(inputs)
    non_blocking = optimizer.evaluate(
        mixed_brake_parameter_template(inputs)
    )
    assert non_blocking < PENALTY_MAX_BLOCKING

    grid = BrakeGridSettings(
        brake_min_deg=inputs.brake_min_deg,
        brake_max_deg=inputs.brake_max_deg,
        position_count=5,
    )
    motion_angles = tuple(angle for angle in _motion(inputs))
    target = _blended_targets(
        inputs,
        grid.fractions,
        input_angles=motion_angles,
    )[0]
    good = SimulationResult(
        input_angles=motion_angles[:2],
        output_angles=(
            motion_angles[0],
            motion_angles[1],
        ),
        success=True,
    )
    blocked = SimulationResult(
        input_angles=motion_angles[:1],
        output_angles=(motion_angles[0],),
        success=False,
        blocked_at=motion_angles[1],
    )
    fitness = BrakeCurveFitness()
    non_blocking_fitness = fitness.evaluate([(good, target)])
    blocking_fitness = fitness.evaluate(
        [(good, target), (blocked, target)]
    )
    assert blocking_fitness == PENALTY_MAX_BLOCKING
    assert blocking_fitness > non_blocking_fitness
    assert non_blocking < blocking_fitness


def test_brake_travel_reduced_to_zero_when_braked_blocks():
    """The start mechanism is non-blocking at the unbraked
    position while the braked position blocks; the generated
    solution must reduce the brake lever travel to 0 deg
    (angle_max clamped onto angle_start), so reloading it can
    never generate a blocking braked solution (issue #115)."""
    inputs = _blocking_inputs()
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
        position_count=5,
    )
    assert _braked_blocks_but_unbraked_works(grid)
    reduced = _reduce_brake_travel_to_zero(inputs.definition)
    brake = reduced.get_lever(_brake_lever_id(reduced))
    assert brake.angle_max == pytest.approx(
        brake.angle_start
    )
    assert brake.angle_max == pytest.approx(0.0)
    # Reloading the reduced solution (as the GUI does) yields
    # a zero-travel brake range in which every brake position
    # coincides with the non-blocking unbraked one.
    reduced_inputs = mixed_brake_inputs_from_parts(
        reduced,
        inputs.unbraked,
        inputs.braked,
    )
    assert reduced_inputs.brake_max_deg == (
        pytest.approx(reduced_inputs.brake_min_deg)
    )
    reduced_grid = build_mixed_brake_grid(
        reduced_inputs,
        reduced,
        position_count=5,
    )
    assert all(
        state is not None for state in reduced_grid.states
    )


def test_optimization_never_generates_blocking_braked_solution():
    """End-to-end invariant (issue #115): whenever the
    generated solution is non-blocking at the unbraked
    position but blocks at the fully braked position, its
    brake lever travel is 0 deg."""
    inputs = _blocking_inputs()
    result = run_mixed_brake_optimization(
        inputs,
        mixed_brake_parameter_template(inputs),
        _small_settings(),
    )
    assert result.best_score < PENALTY_MAX_BLOCKING
    optimized = result.optimized_definition
    assert optimized is not None
    grid = build_mixed_brake_grid(
        inputs,
        optimized,
        position_count=5,
    )
    brake = optimized.get_lever(_brake_lever_id(optimized))
    if _braked_blocks_but_unbraked_works(grid):
        assert brake.angle_max == pytest.approx(
            brake.angle_start
        )


def test_generated_solution_keeps_travel_when_braked_buildable():
    """When the braked position builds and simulates fine,
    the brake lever travel of the generated solution stays
    untouched (no over-triggering of the 0 deg clamp)."""
    inputs = _example_inputs()
    result = run_mixed_brake_optimization(
        inputs,
        mixed_brake_parameter_template(inputs),
        _small_settings(),
    )
    optimized = result.optimized_definition
    assert optimized is not None
    brake = optimized.get_lever(_brake_lever_id(optimized))
    assert brake.angle_max == pytest.approx(
        math.radians(190.0)
    )
    assert brake.angle_max > brake.angle_start
