"""\
tests/test_issue113_target_curves_displayed.py

Regression test for issue #113: "Mechanism blocked while
Fitness is below 1".

The mixed brake optimization skips brake positions at which
the mechanism cannot be built when scoring a candidate
(issue #111), so the best candidate can reach a fitness
below 1 even though it is only buildable at a subset of the
brake positions (for the attached optimized mechanism only
the unbraked 0 % position).  The result display then failed:
the report omitted the target curves of the unbuildable
positions entirely and the GUI result tab showed only the
"blockiert" error for the selected brake position, so no
target curve could be displayed at all.

With the fix, the report always draws the blended target
curve of EVERY brake position (the target exists
independently of the mechanism) and the actual curve of the
buildable ones, and the GUI falls back to this overall
report when the selected single position cannot be built.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import pytest

from analysis.brake_curve_fitness import (
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
    _blended_targets,
    _endpoint_constraint,
    _motion,
    _reference_output_angle,
    build_mixed_brake_grid,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
    mixed_brake_preview,
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

APP_FILE = (
    Path(__file__).parent.parent / "gui" / "app.py"
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


def _inputs():
    return load_mixed_brake_inputs(
        MECHANISM_CSV,
        UNBRAKED_CSV,
        BRAKED_CSV,
    )


def _plot_data(html: str) -> dict:
    match = re.search(
        r'<script id="plot-data" '
        r'type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


def _optimizer(inputs):
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


def test_optimized_mechanism_only_buildable_at_zero_percent():
    """Documents the issue's situation: the optimized
    mechanism from the attachment can only be built at the
    unbraked 0 % brake position, yet the optimizer scores
    it below 1 (unbuildable positions are skipped, #111)."""
    inputs = _inputs()
    grid = build_mixed_brake_grid(
        inputs,
        inputs.definition,
        position_count=5,
    )
    assert grid.states[0] is not None
    assert all(
        state is None for state in grid.states[1:]
    )
    optimizer, template = _optimizer(inputs)
    assert optimizer.evaluate(template) < 1.0


def test_report_shows_target_curves_of_all_brake_positions():
    """Despite the mechanism being buildable only at 0 %,
    the report must draw the blended target curve of EVERY
    brake position plus the actual curve of the buildable
    one (issue #113: no target curve may disappear)."""
    inputs = _inputs()
    html = mixed_brake_preview(
        inputs,
        mixed_brake_parameter_template(inputs),
    )
    payload = _plot_data(html)
    labels = [
        series["label"] for series in payload["curves"]
    ]
    for pct in (0, 25, 50, 75, 100):
        assert f"Soll Bremse {pct} %" in labels
    assert "Bremse 0 %" in labels
    assert "Bremse 100 %" not in labels
    assert payload["mechanism_states"]


def test_result_tab_falls_back_to_full_report():
    """gui/app.py is a top-level Streamlit script and cannot
    be imported, so inspect its AST: the mixed result tab's
    ``if result_state is not None:`` branch must have an
    ``else`` that renders the overall ``mixed_result.html``
    report when the selected brake position cannot be built
    (issue #113)."""
    tree = ast.parse(APP_FILE.read_text(encoding="utf-8"))
    fallback = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "result_state"
            and any(
                isinstance(op, ast.IsNot)
                for op in test.ops
            )
        ):
            continue
        for stmt in node.orelse:
            if isinstance(stmt, ast.Expr) and isinstance(
                stmt.value, ast.Call
            ):
                call = stmt.value
                func = call.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "iframe"
                    and call.args
                    and isinstance(call.args[0], ast.Attribute)
                    and call.args[0].attr == "html"
                    and isinstance(
                        call.args[0].value, ast.Name
                    )
                    and call.args[0].value.id == "mixed_result"
                ):
                    fallback = call
    assert fallback is not None, (
        "the mixed result tab must fall back to "
        "st.iframe(mixed_result.html, ...) when the "
        "selected brake position cannot be built (issue #113)"
    )
