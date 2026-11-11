"""\
tests/test_issue111_mixed_curve_quality.py

Regression test for issue #111: "Quality of mixed curve
worse than single curve".

The issue loads the IDENTICAL target curve for the braked
and the unbraked base curve in mixed mode and starts from a
mechanism that is already a very good fit of the single-curve
optimization (a brake lever was inserted so that at the
initial brake position the mixer lever position is identical
to the single-curve mechanism).

Root cause: ``MixedBrakeOptimizer.evaluate`` returned the
flat hard penalty ``PENALTY_MAX_BLOCKING`` as soon as ONE
brake position could not be built.  For the issue's
mechanism every brake position beta != 0 is geometrically
unreachable, so EVERY candidate tied at exactly 100000: the
fitness landscape was flat, the optimizer had no gradient
and "found only blocking solutions", while the same initial
mechanism scores ~0.03 in single mode.

With the fix, unbuildable brake positions are skipped and the
remaining positions are still scored, so the initial
candidate keeps its single-curve quality in mixed mode.
"""

from __future__ import annotations

import math

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
    _endpoint_constraint,
    _motion,
    _reference_output_angle,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
    run_mixed_brake_optimization,
)
from gui.workflow import (
    OptimizationSettings,
    _build_optimizer,
    load_inputs,
    parameter_template,
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

MECHANISM_SINGLE_CSV = """\
id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on
1;50;100;73,05799607;60;300;133,651956;0;0;239;1;0;0;;;;;;
2;50;100;76,9929424;100;260;153,1432978;0;0;-261;1;0;0;;;;1;;
3;50;100;91,73216337;-90;90;11,8383644;0;0;-261;1;0;0;;;;;2;
4;50;100;62,09694189;-70;70;-5,795825305;0;-40;-1020;1;0;0;;;;3;;
5;50;100;92,07312009;60;300;147,4731758;-815;-40;-1020;1;0;0;;;;;4;
6;70;85;82;30;130;80;-850;-218;-4020;0;1;0;;;;5;;
"""

MECHANISM_MIXED_CSV = """\
id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on
1;50;100;73,05799607;60;300;133,651956;0;0;239;1;0;0;;;;;;
2;50;100;76,9929424;100;260;153,1432978;0;0;-261;1;0;0;;;;1;;7
3;50;100;91,73216337;-90;90;11,8383644;0;0;-261;1;0;0;;;;;2;7
4;50;100;62,09694189;-70;70;-5,795825305;0;-40;-1020;1;0;0;;;;3;;
5;50;100;92,07312009;60;300;147,4731758;-815;-40;-1020;1;0;0;;;;;4;
6;70;85;82;30;130;80;-850;-218;-4020;0;1;0;;;;5;;
7;2000;2050;2000;-10;10;0;0;-2000;-261;1;0;0;;;;;;
"""

TARGET_CSV = """\
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


def _single_mode_fitness() -> float:
    """Single-curve fitness of the initial single mechanism."""
    inputs = load_inputs(
        MECHANISM_SINGLE_CSV,
        TARGET_CSV,
    )
    template = parameter_template(inputs)
    evaluator = _build_optimizer(
        inputs,
        OptimizationSettings(workers=1),
    )
    return evaluator(template)


def _mixed_inputs():
    return load_mixed_brake_inputs(
        MECHANISM_MIXED_CSV,
        TARGET_CSV,
        TARGET_CSV,
    )


def _mixed_optimizer(inputs):
    template = mixed_brake_parameter_template(inputs)
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
    return (
        MixedBrakeOptimizer(
            builder=MixedBrakeBuilder(inputs.definition),
            simulator=MechanismSimulator(motion=motion),
            fitness=BrakeCurveFitness(
                weights=tuple(
                    1.0 for _ in brake_positions
                ),
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
        ),
        template,
    )


def test_initial_candidate_keeps_single_curve_quality():
    """The initial mixed mechanism (single-curve optimum plus
    an inserted brake lever, same target curve for both base
    curves) must NOT collapse onto the blocking penalty in
    mixed mode; its fitness stays in the same range as the
    single-curve fitness."""
    single = _single_mode_fitness()
    assert single < 1.0

    inputs = _mixed_inputs()
    optimizer, template = _mixed_optimizer(inputs)
    mixed = optimizer.evaluate(template)
    assert mixed < PENALTY_MAX_BLOCKING
    assert mixed < 1.0
    assert mixed == pytest.approx(single, abs=0.1)


def test_short_mixed_optimization_not_blocking_only():
    """A short mixed-mode optimization over the issue's files
    must reach a finite, non-blocking best score instead of
    finding only blocking solutions."""
    inputs = _mixed_inputs()
    template = mixed_brake_parameter_template(inputs)
    settings = MixedBrakeSettings(
        brake_positions=5,
        population_size=20,
        children_count=20,
        selection_count=6,
        max_generations=10,
        workers=1,
    )
    result = run_mixed_brake_optimization(
        inputs,
        template,
        settings,
    )
    assert result.best_score < PENALTY_MAX_BLOCKING
    assert result.best_score < 1.0
