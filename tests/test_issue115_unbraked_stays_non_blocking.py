"""\
tests/test_issue115_unbraked_stays_non_blocking.py

Regression test for the re-opened issue #115 ("non-blocking
solution for unbreaked curve must never lead to blocking
solution of breaked curve") with the example files attached
to the issue comment.

The attached input mechanism (mechanism_optimized_mixed2.csv)
is non-blocking at the unbraked 0 % brake position, but its
fully braked 100 % position blocks during simulation.  The
fixed version nevertheless generated a solution
(mechanism_optimized_blocksAtUnbreakedPosition.csv) whose
brake lever start angle moved so far that the mechanism
blocks at the UNBRAKED position while the braked position is
non-blocking.  The algorithm shall keep the unbraked state
non-blocking once this is achieved.

Root cause: the issue #111 fix skipped EVERY unbuildable
brake position, including the unbraked one, so a candidate
that cannot be built at the unbraked position was still
scored on the braked positions and could win.

With the fix, the unbraked (first) brake position is never
skipped: a build error there gives the candidate the hard
penalty ``PENALTY_MAX_BLOCKING``, so a blocking-at-unbraked
candidate always scores worse than a non-blocking one and
can never become the generated solution.
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
    _braked_blocks_but_unbraked_works,
    _brake_lever_id,
    _endpoint_constraint,
    _motion,
    _reference_output_angle,
    build_mixed_brake_grid,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
    run_mixed_brake_optimization,
)
from mechanics.mixed_brake_builder import (
    MixedBrakeBuilder,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.mixed_brake_optimizer import (
    MixedBrakeOptimizer,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)

MECHANISM_CSV = """\
id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on
1;50;100;73,05799607;95;300;133,651956;0;0;239;1;0;0;;;;;;
2;50;100;76,9929424;100;260;153,1432978;0;0;-261;1;0;0;;;;1;;7
3;50;100;91,73216337;-90;90;11,8383644;0;0;-261;1;0;0;;;;;2;7
4;50;100;62,09694189;-70;70;-5,795825305;0;-40;-1020;1;0;0;;;;3;;
5;50;100;92,07312009;60;300;147,4731758;-815;-40;-1020;1;0;0;;;;;4;
6;70;85;82;30;130;80;-850;-218;-4020;0;1;0;;;;5;;
7;10;100;10;-180;360;270;0;0;-251;1;0;0;;;;;;
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

# The solution the fixed version wrongly generated: its brake
# lever start angle (357.5 deg) makes the unbraked position
# unbuildable while the braked 360 deg position works.
BLOCKS_AT_UNBRAKED_CSV = """\
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,ref_x,ref_y,ref_z,driver,coupled,pivot_on
1,50.0,100.0,71.83594147988023,95.0,300.0,107.29422774698804,0.0,0.0,239.0,1.0,0.0,0.0,,,,,,
2,50.0,100.0,86.93135412365879,100.0,260.0,104.11997274233552,0.0,0.0,-261.0,1.0,0.0,0.0,,,,1,,7
3,50.0,100.0,94.3222781979507,-90.0,90.0,0.9113422407981492,0.0,0.0,-261.0,1.0,0.0,0.0,,,,,2,7
4,50.0,100.0,59.16722207397132,-70.0,70.0,-38.111461151357396,0.0,-40.0,-1020.0,1.0,0.0,0.0,,,,3,,
5,50.0,100.0,67.27060670866197,59.99999999999999,300.0,176.65037905649373,-815.0,-40.0,-1020.0,1.0,0.0,0.0,,,,,4,
6,70.0,85.0,82.0,29.999999999999996,130.0,80.0,-850.0,-218.0,-4020.0,0.0,1.0,0.0,,,,5,,
7,10.0,100.0,43.71792550705189,-180.0,360.0,357.5469924230896,0.0,0.0,-251.0,1.0,0.0,0.0,,,,,,
"""


def _inputs():
    return load_mixed_brake_inputs(
        MECHANISM_CSV,
        UNBRAKED_CSV,
        BRAKED_CSV,
    )


def _optimizer(inputs) -> MixedBrakeOptimizer:
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


def _blocks_at_unbraked_inputs():
    return load_mixed_brake_inputs(
        BLOCKS_AT_UNBRAKED_CSV,
        UNBRAKED_CSV,
        BRAKED_CSV,
    )


def test_attached_mechanism_is_non_blocking_at_unbraked():
    """Documents the issue's premise: a non-blocking solution
    for the unbraked curve exists (the attached input
    mechanism builds and simulates fine at 0 %; its fully
    braked 100 % position blocks)."""
    inputs = _inputs()
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
        position_count=5,
    )
    assert grid.states[0] is not None
    assert grid.states[-1] is None


def _unbraked_only_optimizer(inputs) -> MixedBrakeOptimizer:
    """Optimizer over the unbraked position only, used to
    score the non-blocking unbraked solution of the attached
    input mechanism."""
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
        fitness=BrakeCurveFitness(),
        definition=inputs.definition,
        brake_positions=brake_positions[:1],
        targets=targets[:1],
    )


def test_candidate_blocking_at_unbraked_scores_worse():
    """The solution the fixed version generated moves the
    brake lever start angle to 357.5 deg.  Since issue #118
    the unbraked position follows from the candidate's own
    brake lever angle range, so this candidate is evaluated
    at 357.5 deg (where it builds, because the reference rods
    are frozen there) and at the braked 360 deg.  Its curves
    block during simulation, so it must still score at
    least the hard penalty and strictly worse than a
    non-blocking solution of the unbraked curve (issue
    #115)."""
    inputs = _inputs()
    blocks_inputs = _blocks_at_unbraked_inputs()
    blocking_grid = build_mixed_brake_grid(
        blocks_inputs,
        blocks_inputs.definition,
        position_count=5,
    )
    assert blocking_grid.states[0] is None
    assert blocking_grid.states[-1] is not None

    optimizer = _optimizer(inputs)
    template = mixed_brake_parameter_template(inputs)
    non_blocking = _unbraked_only_optimizer(inputs).evaluate(
        template
    )
    assert non_blocking < PENALTY_MAX_BLOCKING
    blocking_candidate = CsvParameterFactory.create(
        blocks_inputs.definition,
    )
    blocking = optimizer.evaluate(blocking_candidate)
    assert blocking >= PENALTY_MAX_BLOCKING
    assert blocking > non_blocking


def test_optimization_keeps_unbraked_non_blocking():
    """End-to-end: the generated solution must keep the
    unbraked state non-blocking (issue #115); rebuilding the
    result grid from the optimized definition must yield a
    buildable, non-blocking 0 % position."""
    inputs = _inputs()
    result = run_mixed_brake_optimization(
        inputs,
        mixed_brake_parameter_template(inputs),
        MixedBrakeSettings(
            population_size=8,
            children_count=8,
            selection_count=4,
            max_generations=2,
            brake_positions=5,
            workers=1,
        ),
    )
    assert result.best_score < PENALTY_MAX_BLOCKING
    optimized = result.optimized_definition
    assert optimized is not None
    # Rebuild with the original inputs so the motion stays
    # anchored exactly as during the evaluation of the
    # candidate (the reload path re-anchors at the optimized
    # drive angle, which is a separate concern).
    result_grid = build_mixed_brake_grid(
        inputs,
        optimized,
        position_count=5,
    )
    assert result_grid.states[0] is not None
