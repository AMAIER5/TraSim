"""
tests/test_issue71_false_positive_blocking.py

Regression test for issue #71: the optimization never
reached a fitness below 10000 for a mechanism that is
clearly not blocked over the target motion.

Root cause: ``CurveFitness`` treated every validation
failure as a block.  The stage validation sweeps the
stage's complete build-space lever-angle segment, which
is typically much wider than the input interval the
stage actually traverses during the target motion, so
failures far outside the traversed interval were
reported as blocking although the motion never reaches
those configurations.

The mechanism below is the one from the issue (8 levers,
4 stages).  With the fix, the start configuration scores
well below PENALTY_BASE.
"""

from __future__ import annotations

import math

from analysis.curve_fitness import (
    CurveFitness,
    PENALTY_BASE,
)
from analysis.target_curve import TargetCurve
from mechanism_io.csv_reader import CsvReader
from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.point_motion import PointMotion
from simulation.stage_simulator import StageSimulator


MECHANISM_CSV = """\
id;length_min;length_max;length_start;angle_min;angle_max;angle_start;pivot_x;pivot_y;pivot_z;axis_x;axis_y;axis_z;ref_x;ref_y;ref_z;driver;coupled;pivot_on
1;50,0;100,0;67,47378570809455;59,99999999999999;300,0;158,1354543665006;0,0;0,0;239,0;1,0;0,0;0,0;;;;;
2;50,0;100,0;68,86642872161099;100,0;260,0;167,4407836946384;0,0;0,0;-261,0;1,0;0,0;0,0;;;;1;;
3;50,0;100,0;70,90376712840295;-70,0;70,0;-5,667804376526453;0,0;0,0;-261,0;1,0;0,0;0,0;;;;;2;
4;50,0;100,0;74,00485317494844;-70,0;70,0;-6,453192955968271;0,0;-40,0;-1020,0;1,0;0,0;0,0;;;;3;;
5;50,0;100,0;52,204406220406966;59,99999999999999;300,0;144,20453938644096;-815,0;-40,0;-1020,0;1,0;0,0;0,0;;;;;4;
6;82,0;82,0;82,0;-70,0;70,0;-21,496963828502732;-815,0;-100,0;-1370,0;1,0;0,0;0,0;;;;5;;
7;82,0;82,0;82,0;59,99999999999999;300,0;155,90807695877598;-815,0;-100,0;-1370,0;1,0;0,0;0,0;;;;;6;
8;82,0;82,0;82,0;29,999999999999996;130,0;83,53482653799107;-850,0;-218,0;-4020,0;0,0;1,0;0,0;;;;7;
"""


def _write(
    tmp_path,
    name: str,
    text: str,
):
    path = tmp_path / name
    path.write_text(
        text,
        encoding="utf-8",
    )
    return path


def test_issue71_mechanism_is_not_blocked_over_target_motion(
    tmp_path,
):
    mechanism_path = _write(
        tmp_path,
        "mechanism.csv",
        MECHANISM_CSV,
    )

    definition = CsvReader.read_mechanism(
        mechanism_path,
    )
    builder = CsvMechanismBuilder(definition)

    candidate = CsvParameterFactory.create(
        definition,
    )

    support_deg = (
        145, 150, 155, 160, 170,
        180, 190, 200, 205, 210, 215,
    )
    target_deg = (
        105, 103, 100, 97, 89,
        80, 68, 56, 49, 42, 34,
    )

    drive_angle = (
        definition.levers[0].angle_start
    )

    middle_input = math.radians(
        support_deg[len(support_deg) // 2],
    )

    angles = tuple(
        math.radians(angle) - middle_input + drive_angle
        for angle in support_deg
    )
    outputs = tuple(
        math.radians(angle)
        for angle in target_deg
    )

    target = TargetCurve.from_points_strict(
        angles,
        outputs,
        tuple(1.0 for _ in angles),
    )

    fitness = CurveFitness(
        target_curve=target,
        motion_start=angles[0],
        motion_range=angles[-1] - angles[0],
    )

    simulator = MechanismSimulator(
        motion=PointMotion(angles=angles),
        stage_simulator=StageSimulator(),
        stage_limit=None,
    )

    mechanism = builder.build(candidate)
    simulation = simulator.simulate(mechanism)
    validation = builder.get_validation_results()

    assert all(
        result.success
        for result in simulation
    )

    score = fitness.evaluate(
        tuple(simulation),
        validation,
    )

    assert score < PENALTY_BASE
