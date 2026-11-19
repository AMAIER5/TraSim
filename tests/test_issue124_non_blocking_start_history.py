"""\
tests/test_issue124_non_blocking_start_history.py

Regression test for issue #124 ("Optimization produces
blocked results starting from non-blocking initial
mechanic").

The debug export attached to the issue starts from a
mechanism that is non-blocking at the unbraked 0 % brake
position (the bearing point of the brake lever,
``lever.7.pivot.y`` / ``lever.7.pivot.z``, was freed as an
optimization degree of freedom), yet the exported
``fitness_history`` starts with ``Infinity``: generation 0
was reported as a blocked result although the initial
mechanic is valid.

Root cause: both workflow runners read
``engine.best_score`` for the first history entry directly
after ``engine.evaluate_population()`` but before
``engine.update_best()`` had ever run, so the history always
started with the engine's initial ``inf`` instead of the
best score of the (non-blocking) initial population.

With the fix, the initial population's best is recorded
before it is read, so a non-blocking start mechanism yields
a finite generation-0 fitness.
"""

from __future__ import annotations

import math

import pytest

from analysis.brake_curve_fitness import (
    PENALTY_MAX_BLOCKING,
)
from gui.mixed_brake_workflow import (
    MixedBrakeSettings,
    build_mixed_brake_grid,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
    run_mixed_brake_optimization,
)
from gui.workflow import (
    OptimizationSettings,
    load_inputs,
    parameter_template,
    run_optimization,
)
from optimization.parameter import Parameter
from optimization.parameter_set import ParameterSet

# Initial mechanism of the debug export attached to issue
# #124 (mixed_brake mode, 7 levers; levers 2 and 3 pivot on
# the brake lever 7).
MECHANISM_CSV = """\
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,ref_x,ref_y,ref_z,driver,coupled,pivot_on
1,50.0,100.0,97.93844899883378,95.0,300.0,98.32761238859081,0.0,0.0,239.0,1.0,0.0,0.0,,,,,,
2,50.0,100.0,99.58050760061515,100.0,260.0,140.80859411632906,0.0,0.0,-261.0,1.0,0.0,0.0,,,,1,,7
3,50.0,100.0,91.405415321314,-90.0,90.0,-14.81295473920281,0.0,0.0,-261.0,1.0,0.0,0.0,,,,,2,7
4,50.0,100.0,80.63367539777822,-70.0,70.0,-8.012730973426944,0.0,-40.0,-1020.0,1.0,0.0,0.0,,,,3,,
5,50.0,100.0,79.98880295657216,59.99999999999999,300.0,169.95854151876338,-815.0,-40.0,-1020.0,1.0,0.0,0.0,,,,,4,,
6,70.0,85.0,82.0,29.999999999999996,200.0,80.0,-850.0,-218.0,-4020.0,0.0,1.0,0.0,,,,5,,
7,500.0,600.0,523.1513504276427,170.0,190.0,177.02428309084652,0.0,301.8584004820778,-333.17851411047855,1.0,0.0,0.0,,,,,,
"""

# Target curves of the debug export (with per-point weights).
UNBRAKED_CSV = """\
input_angle,output_angle,weight
-35.0,-25.0,3.0
-29.999999999999996,-23.0,1.0
-25.0,-20.0,1.0
-20.0,-17.0,1.0
-10.0,-9.0,1.0
0.0,0.0,5.0
10.0,12.000000000000002,1.0
20.0,24.000000000000004,1.0
25.0,31.0,1.0
29.999999999999996,38.0,1.0
35.0,46.0,3.0
"""

BRAKED_CSV = """\
input_angle,output_angle,weight
-35.0,19.0,1.0
-29.999999999999996,27.0,1.0
-25.0,34.0,1.0
-20.0,40.0,1.0
-10.0,50.0,1.0
0.0,59.99999999999999,1.0
10.0,68.0,1.0
20.0,76.0,1.0
25.0,79.0,2.0
29.999999999999996,82.0,2.0
35.0,85.0,1.0
"""

SIMPLE_MECHANISM_CSV = """\
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,50,150,50,30,150,90,0,0,0,0,0,1,,2
2,50,150,60,30,150,90,100,0,0,0,0,1,1,
"""

SIMPLE_TARGET_CSV = """\
input_angle,output_angle
60,90
70,100
80,110
90,120
100,130
110,140
120,150
"""


def _mixed_inputs():
    return load_mixed_brake_inputs(
        MECHANISM_CSV,
        UNBRAKED_CSV,
        BRAKED_CSV,
    )


def _template_with_bearing_point_dofs(inputs):
    """Parameter template of the issue run: the bearing
    point of the brake lever (lever 7 pivot y/z) is a free
    optimization degree of freedom, exactly as in the debug
    export's ``optimizable_parameters``."""
    template = mixed_brake_parameter_template(inputs)
    parameters = []
    for parameter in template.parameters:
        if parameter.name == "lever.7.pivot.y":
            parameter = Parameter(
                name=parameter.name,
                minimum=250.0,
                maximum=600.0,
                value=parameter.value,
            )
        elif parameter.name == "lever.7.pivot.z":
            parameter = Parameter(
                name=parameter.name,
                minimum=-400.0,
                maximum=-150.0,
                value=parameter.value,
            )
        parameters.append(parameter)
    return ParameterSet(tuple(parameters))


def test_initial_mechanism_is_non_blocking_at_unbraked():
    """Premise of issue #124: the start mechanism of the
    debug export builds and simulates at the unbraked 0 %
    brake position (it is only unbuildable at higher brake
    positions)."""
    inputs = _mixed_inputs()
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
        position_count=5,
    )
    assert grid.states[0] is not None


def test_mixed_history_starts_finite_for_non_blocking_start():
    """The fitness history must start with the real best
    score of the initial population, not with the blocking
    infinity: starting from the non-blocking initial
    mechanism of the issue, generation 0 is a valid
    (finite) result (issue #124)."""
    inputs = _mixed_inputs()
    template = _template_with_bearing_point_dofs(inputs)
    settings = MixedBrakeSettings(
        population_size=12,
        children_count=8,
        selection_count=4,
        max_generations=2,
        brake_positions=5,
        workers=1,
        seed=42,
    )
    result = run_mixed_brake_optimization(
        inputs,
        template,
        settings,
    )
    assert math.isfinite(result.fitness_history[0])
    assert (
        result.fitness_history[0]
        < PENALTY_MAX_BLOCKING
    )
    # The first history entry is the initial population's
    # best, identical to the value the engine reports for
    # generation 0 itself.
    assert result.fitness_history[0] == (
        pytest.approx(result.fitness_history[1])
    )
    assert math.isfinite(result.best_score)
    assert result.best_score < PENALTY_MAX_BLOCKING


def test_mixed_result_not_blocked_at_unbraked():
    """The produced result keeps the unbraked position
    buildable and non-blocking (issue #124 in combination
    with #115)."""
    inputs = _mixed_inputs()
    template = _template_with_bearing_point_dofs(inputs)
    settings = MixedBrakeSettings(
        population_size=12,
        children_count=8,
        selection_count=4,
        max_generations=2,
        brake_positions=5,
        workers=1,
        seed=42,
    )
    result = run_mixed_brake_optimization(
        inputs,
        template,
        settings,
    )
    assert result.optimized_definition is not None
    result_grid = build_mixed_brake_grid(
        inputs,
        result.optimized_definition,
        position_count=5,
    )
    assert result_grid.states[0] is not None


def test_single_curve_history_starts_finite():
    """The single-curve workflow has the same history
    initialization; its first entry must also be the real
    (finite) best of the initial population (issue
    #124)."""
    inputs = load_inputs(
        SIMPLE_MECHANISM_CSV,
        SIMPLE_TARGET_CSV,
    )
    settings = OptimizationSettings(
        population_size=10,
        children_count=8,
        selection_count=4,
        max_generations=2,
        workers=1,
        seed=7,
    )
    result = run_optimization(
        inputs,
        parameter_template(inputs),
        settings,
    )
    assert math.isfinite(result.fitness_history[0])
    assert (
        result.fitness_history[0]
        < PENALTY_MAX_BLOCKING
    )
    assert math.isfinite(result.best_score)
