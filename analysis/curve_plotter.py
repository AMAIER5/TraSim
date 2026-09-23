"""analysis/curve_plotter.py

HTML visualization of simulation results.

The plotter produces a self-contained HTML document
with two diagrams:

1. A three-dimensional diagram showing every lever of
   the mechanism as a line in three positions: at the
   minimum, the middle and the maximum drive angle of
   the first (driving) lever.  The coupling rods
   connecting the levers are drawn as lines between
   the lever endpoints; torsion shafts between coupled
   levers are not drawn.
2. A two-dimensional diagram showing the desired
   output curve (Soll) and the achieved output curve
   (Ist) over the drive angle.

All angles are passed in radians (internal unit) and
converted to degrees for display.  Lengths are shown
in millimetres.

The diagrams are rendered with Plotly.js from an
embedded JSON data block; no Python plotting library
is required.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from analysis.target_curve import (
    DiscreteTargetCurve,
    TargetCurve,
)
from core.point3d import Point3D
from mechanics.mechanism import Mechanism

# Line colors used for the levers in the 3D diagram.
# Levers without an explicit color cycle through this
# list.
LEVER_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)

# Line colors used for the actual output curves in the
# 2D diagram.  Curves without an explicit color cycle
# through this list.
CURVE_COLORS: tuple[str, ...] = (
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
)


@dataclass(frozen=True, slots=True)
class LeverPlotLine:
    """
    One lever drawn as a line from pivot to endpoint.

    ``angle`` is the lever angle in radians, measured
    relative to the lever's reference direction.
    """

    name: str
    pivot: Point3D
    end: Point3D
    angle: float
    color: str


@dataclass(frozen=True, slots=True)
class RodPlotLine:
    """
    One coupling rod drawn as a line between the
    endpoints of two levers.
    """

    name: str
    start: Point3D
    end: Point3D
    length: float


@dataclass(frozen=True, slots=True)
class MechanismPlotState:
    """
    Snapshots of all levers of a mechanism at one
    drive angle of the first (driving) lever.

    ``position_label`` describes the meaning of the
    snapshot (``"min"``, ``"mid"`` or ``"max"``),
    ``position_angle`` is the drive angle of the
    first lever in radians.
    """

    position_label: str
    position_angle: float
    levers: tuple[LeverPlotLine, ...]
    rods: tuple[RodPlotLine, ...] = ()

    @classmethod
    def from_mechanism(
        cls,
        mechanism: Mechanism,
        *,
        lever_angles: dict[int, float],
        lever_names: (
            dict[int, str] | None
        ) = None,
        lever_colors: (
            dict[int, str] | None
        ) = None,
        position_label: str = "",
        position_angle: float = 0.0,
    ) -> MechanismPlotState:
        """
        Create a snapshot from a mechanism.

        ``lever_angles`` maps lever ids to lever angles
        in radians; levers without an entry are drawn
        at their stage reference angle.  Lever ids are
        assigned by first appearance in the stage
        chain (input lever of the first stage is 1),
        which matches the CSV lever ids of chain
        mechanisms.
        """
        names = lever_names or {}
        colors = lever_colors or {}
        ordered_levers = _ordered_levers(mechanism)
        levers: list[LeverPlotLine] = []
        endpoints: dict[int, Point3D] = {}
        for index, entry in enumerate(ordered_levers):
            lever_id = index + 1
            angle = lever_angles.get(
                lever_id,
                entry.reference_angle,
            )
            end = entry.lever.end_position(angle)
            endpoints[lever_id] = end
            levers.append(
                LeverPlotLine(
                    name=names.get(
                        lever_id,
                        f"Lever {lever_id}",
                    ),
                    pivot=entry.lever.pivot,
                    end=end,
                    angle=angle,
                    color=colors.get(
                        lever_id,
                        LEVER_COLORS[
                            (lever_id - 1)
                            % len(LEVER_COLORS)
                        ],
                    ),
                ),
            )
        rods = _stage_rods(
            mechanism,
            endpoints,
        )
        return cls(
            position_label=position_label,
            position_angle=position_angle,
            levers=tuple(levers),
            rods=tuple(rods),
        )


@dataclass(frozen=True, slots=True)
class _OrderedLever:
    """
    A lever of the mechanism together with the lever
    angle of its stage reference position.
    """

    lever: Lever
    reference_angle: float


def _ordered_levers(
    mechanism: Mechanism,
) -> tuple[_OrderedLever, ...]:
    """
    Return all distinct levers of a mechanism in stage
    chain order.
    """
    result: list[_OrderedLever] = []
    seen: set[int] = set()
    for stage in mechanism.stages:
        for lever, angle in (
            (
                stage.input_lever,
                stage.input_angle,
            ),
            (
                stage.output_lever,
                stage.output_angle,
            ),
        ):
            if id(lever) in seen:
                continue
            seen.add(id(lever))
            result.append(
                _OrderedLever(
                    lever=lever,
                    reference_angle=angle,
                ),
            )
    return tuple(result)


def _stage_rods(
    mechanism: Mechanism,
    endpoints: dict[int, Point3D],
) -> list[RodPlotLine]:
    """
    Coupling rods of the mechanism for one snapshot.

    Every stage contributes one rod from the endpoint
    of its input lever to the endpoint of its output
    lever.  Torsion shafts between coupled levers are
    not drawn.
    """
    ordered_levers = _ordered_levers(mechanism)
    lever_ids: dict[int, int] = {
        id(entry.lever): index + 1
        for index, entry in enumerate(
            ordered_levers,
        )
    }
    rods: list[RodPlotLine] = []
    for index, stage in enumerate(
        mechanism.stages,
        start=1,
    ):
        input_id = lever_ids[
            id(stage.input_lever)
        ]
        output_id = lever_ids[
            id(stage.output_lever)
        ]
        rods.append(
            RodPlotLine(
                name=(
                    f"Koppelstange {index}"
                ),
                start=endpoints[input_id],
                end=endpoints[output_id],
                length=stage.rod_length,
            ),
        )
    return rods


@dataclass(frozen=True, slots=True)
class CurvePlotSeries:
    """
    One input/output angle series of the 2D diagram.

    All angles are in radians and are converted to
    degrees when the HTML document is written.
    """

    input_angles: tuple[float, ...]
    output_angles: tuple[float, ...]
    label: str
    color: str


class CurvePlotter:
    """
    Collects visualization data and writes an HTML
    document with two diagrams.

    Usage
    -----
    1. Add the mechanism states (three positions of
       all levers) with ``add_mechanism_state``.
    2. Add the target curve and one or more actual
       curves with ``set_target_curve`` and
       ``add_actual_curve``.
    3. Write the document with ``write``.
    """

    def __init__(
        self,
        *,
        title: str = "TraSim \u2014 Ergebnis",
    ) -> None:
        self._title = title
        self._mechanism_states: list[
            MechanismPlotState
        ] = []
        self._target_curve: (
            CurvePlotSeries | None
        ) = None
        self._actual_curves: list[
            CurvePlotSeries
        ] = []

    def add_mechanism_state(
        self,
        state: MechanismPlotState,
    ) -> None:
        """
        Add one mechanism snapshot to the 3D diagram.
        """
        self._mechanism_states.append(state)

    def set_target_curve(
        self,
        target: (
            TargetCurve
            | DiscreteTargetCurve
        ),
        *,
        input_angles: (
            tuple[float, ...] | None
        ) = None,
        output_angles: (
            tuple[float, ...] | None
        ) = None,
    ) -> None:
        """
        Set the desired output curve (Soll).

        Accepts an interpolating ``TargetCurve`` or a
        non-interpolating ``DiscreteTargetCurve``.  The
        support points of the curve are plotted; for a
        ``TargetCurve`` built with ``from_points`` or
        ``from_csv`` they are recovered automatically,
        alternatively they can be passed explicitly
        with ``input_angles``/``output_angles``.
        """
        if (
            input_angles is None
            or output_angles is None
        ):
            if isinstance(
                target,
                DiscreteTargetCurve,
            ):
                input_angles = (
                    target.input_angles
                )
                output_angles = (
                    target.output_angles
                )
            else:
                input_angles = (
                    target_input_angles(target)
                )
                sampled = target.sample(
                    input_angles,
                )
                output_angles = (
                    sampled.output_angles
                )
        self._target_curve = CurvePlotSeries(
            input_angles=input_angles,
            output_angles=output_angles,
            label="Soll-Ausgangskurve",
            color="#2ca02c",
        )

    def add_actual_curve(
        self,
        input_angles: tuple[float, ...],
        output_angles: tuple[float, ...],
        *,
        label: str = "Ist-Ausgangskurve",
        color: str | None = None,
    ) -> None:
        """
        Add one achieved output curve (Ist).
        """
        index = len(self._actual_curves)
        self._actual_curves.append(
            CurvePlotSeries(
                input_angles=input_angles,
                output_angles=output_angles,
                label=label,
                color=color
                or CURVE_COLORS[
                    index % len(CURVE_COLORS)
                ],
            ),
        )

    # ---------------------------------------------------------
    # HTML generation
    # ---------------------------------------------------------

    def build_data(self) -> dict:
        """
        Return the visualization data as a JSON
        serializable dictionary.  All angles are
        converted to degrees.
        """
        mechanism_states: list[dict] = []
        for state in self._mechanism_states:
            levers = [
                {
                    "name": lever.name,
                    "pivot": _point(
                        lever.pivot,
                    ),
                    "end": _point(lever.end),
                    "angle": math.degrees(
                        lever.angle,
                    ),
                    "color": lever.color,
                }
                for lever in state.levers
            ]
            rods = [
                {
                    "name": rod.name,
                    "start": _point(rod.start),
                    "end": _point(rod.end),
                    "length": rod.length,
                }
                for rod in state.rods
            ]
            mechanism_states.append(
                {
                    "position_label": (
                        state.position_label
                    ),
                    "position_angle": (
                        math.degrees(
                            state.position_angle,
                        )
                    ),
                    "levers": levers,
                    "rods": rods,
                }
            )
        return {
            "title": self._title,
            "mechanism_states": mechanism_states,
            "curves": [
                _series_payload(series)
                for series in (
                    self._curve_series()
                )
                if series is not None
            ],
        }

    def _curve_series(
        self,
    ) -> list[CurvePlotSeries | None]:
        return [
            self._target_curve,
            *self._actual_curves,
        ]

    def build_html(self) -> str:
        """
        Return the complete HTML document.
        """
        payload = self.build_data()
        return _HTML_TEMPLATE.format(
            title=_escape(self._title),
            data_json=json.dumps(
                payload,
                ensure_ascii=False,
            ),
        )

    def write(
        self,
        filename: str | Path,
    ) -> Path:
        """
        Write the HTML document to disk.
        """
        path = Path(filename)
        parent = path.parent
        if str(parent):
            parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        path.write_text(
            self.build_html(),
            encoding="utf-8",
        )
        return path


def target_input_angles(
    target: TargetCurve,
) -> tuple[float, ...]:
    """
    Extract the support input angles of a target curve
    created with ``TargetCurve.from_points`` or
    ``TargetCurve.from_csv``.

    The input angles are recovered from the closure of
    the interpolation function.
    """
    closure = getattr(
        target.function,
        "__closure__",
        None,
    )
    if closure is not None:
        for cell in closure:
            value = cell.cell_contents
            if (
                isinstance(value, tuple)
                and len(value) >= 2
                and isinstance(
                    value[0],
                    float,
                )
            ):
                return value
    raise ValueError(
        "Cannot extract support points from the "
        "target curve; pass input_angles and "
        "output_angles explicitly."
    )


def mechanism_states_from_results(
    mechanism: Mechanism,
    results: tuple,
    *,
    position_labels: (
        tuple[str, str, str]
    ) = ("min", "mid", "max"),
) -> tuple[
    MechanismPlotState,
    MechanismPlotState,
    MechanismPlotState,
]:
    """
    Create the three standard snapshots (minimum, middle
    and maximum drive angle of the first lever) from a
    complete stage simulation.

    ``results`` is the tuple returned by
    ``MechanismSimulator.simulate`` for a chain mechanism:
    stage ``i`` connects lever ``i`` with lever ``i + 1``,
    so the output angles of stage ``i`` are the lever
    angles of lever ``i + 1``.  The three positions are
    taken from the first sample, the sample closest to
    the middle of the drive range and the last sample
    of the simulation.

    Raises ``ValueError`` when the simulation contains no
    samples.
    """
    if not results:
        raise ValueError(
            "No simulation results available."
        )
    input_angles = results[0].input_angles
    if not input_angles:
        raise ValueError(
            "Simulation contains no samples; the "
            "mechanism is blocked."
        )
    first = 0
    last = len(input_angles) - 1
    middle_angle = (
        input_angles[first]
        + input_angles[last]
    ) / 2.0
    middle = min(
        range(len(input_angles)),
        key=lambda index: abs(
            input_angles[index] - middle_angle
        ),
    )
    states: list[MechanismPlotState] = []
    for label, sample in (
        (position_labels[0], first),
        (position_labels[1], middle),
        (position_labels[2], last),
    ):
        lever_angles: dict[int, float] = {}
        for stage_index, result in enumerate(
            results,
        ):
            if stage_index == 0:
                lever_angles[1] = (
                    result.input_angles[
                        sample
                    ]
                )
            lever_angles[
                stage_index + 2
            ] = result.output_angles[
                sample
            ]
        states.append(
            MechanismPlotState.from_mechanism(
                mechanism,
                lever_angles=lever_angles,
                position_label=label,
                position_angle=lever_angles[1],
            ),
        )
    return (
        states[0],
        states[1],
        states[2],
    )


def _point(
    point: Point3D,
) -> dict:
    return {
        "x": point.x,
        "y": point.y,
        "z": point.z,
    }


def _series_payload(
    series: CurvePlotSeries,
) -> dict:
    return {
        "label": series.label,
        "color": series.color,
        "input_angles": [
            math.degrees(angle)
            for angle in series.input_angles
        ],
        "output_angles": [
            math.degrees(angle)
            for angle in series.output_angles
        ],
    }


def _escape(
    text: str,
) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
body {{
    font-family: "Segoe UI", Arial, sans-serif;
    margin: 0;
    padding: 24px;
    background: #f5f6f8;
    color: #1f2937;
}}
h1 {{
    font-size: 1.5rem;
    margin: 0 0 4px 0;
}}
h2 {{
    font-size: 1.15rem;
    margin: 28px 0 8px 0;
}}
p {{
    margin: 0 0 12px 0;
    color: #4b5563;
}}
.plot-container {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 8px;
}}
#lever-plot {{
    width: 100%;
    height: 620px;
}}
#curve-plot {{
    width: 100%;
    height: 520px;
}}
</style>
</head>
<body>
<h1 id="title"></h1>
<p>
Hebeldarstellung in drei Positionen (minimaler / mittlerer /
maximaler Antriebswinkel des ersten Hebels) mit Koppelstangen
sowie Soll- und Ist-&Uuml;bertragungskurve.
Winkel in Grad, L&auml;ngen in Millimetern.
</p>

<h2>Hebel und Koppelstangen in drei Positionen (3D)</h2>
<div class="plot-container">
    <div id="lever-plot"></div>
</div>

<h2>Soll- und Ist-Kurve</h2>
<div class="plot-container">
    <div id="curve-plot"></div>
</div>

<script id="plot-data" type="application/json">{data_json}</script>
<script>
"use strict";

const data = JSON.parse(
    document.getElementById("plot-data").textContent
);

document.getElementById(
    "title"
).textContent = data.title;

const POSITION_DASH = {{
    "min": "solid",
    "mid": "dot",
    "max": "dash"
}};

const ROD_COLOR = "#7f7f7f";

function leverTraces() {{
    const traces = [];
    for (const state of data.mechanism_states) {{
        const dash = POSITION_DASH[state.position_label] || "solid";
        for (const rod of state.rods || []) {{
            traces.push({{
                type: "scatter3d",
                mode: "lines",
                x: [rod.start.x, rod.end.x],
                y: [rod.start.y, rod.end.y],
                z: [rod.start.z, rod.end.z],
                name: rod.name + " (" + state.position_label + ")",
                legendgroup: rod.name,
                showlegend: state.position_label === "min",
                line: {{
                    color: ROD_COLOR,
                    width: 6,
                    dash: dash
                }},
                hovertemplate: rod.name +
                    "<br>L&auml;nge: " + rod.length.toFixed(1) + " mm" +
                    "<extra></extra>"
            }});
        }}
        for (const lever of state.levers) {{
            traces.push({{
                type: "scatter3d",
                mode: "lines",
                x: [lever.pivot.x, lever.end.x],
                y: [lever.pivot.y, lever.end.y],
                z: [lever.pivot.z, lever.end.z],
                name: lever.name + " (" + state.position_label + ")",
                legendgroup: lever.name,
                showlegend: state.position_label === "min",
                line: {{
                    color: lever.color,
                    width: 10,
                    dash: dash
                }},
                hovertemplate: lever.name +
                    "<br>Position: " + state.position_label +
                    "<br>Winkel: " + lever.angle.toFixed(1) + " &deg;" +
                    "<extra></extra>"
            }});
            traces.push({{
                type: "scatter3d",
                mode: "markers",
                x: [lever.pivot.x],
                y: [lever.pivot.y],
                z: [lever.pivot.z],
                name: lever.name + " pivot",
                legendgroup: lever.name,
                showlegend: false,
                marker: {{
                    color: lever.color,
                    size: 5,
                    symbol: "circle"
                }},
                hovertemplate: "Drehpunkt " + lever.name +
                    "<extra></extra>"
            }});
        }}
    }}
    return traces;
}}

function leverLayout() {{
    return {{
        margin: {{ t: 16, r: 16, b: 16, l: 16 }},
        legend: {{ itemsizing: "constant" }},
        scene: {{
            aspectmode: "data",
            xaxis: {{ title: "X [mm]" }},
            yaxis: {{ title: "Y [mm]" }},
            zaxis: {{ title: "Z [mm]" }}
        }}
    }};
}}

function curveTraces() {{
    const traces = [];
    for (const series of data.curves) {{
        const isTarget = series.label.indexOf("Soll") === 0;
        traces.push({{
            type: "scatter",
            mode: isTarget ? "lines+markers" : "lines+markers",
            x: series.input_angles,
            y: series.output_angles,
            name: series.label,
            line: {{
                color: series.color,
                width: 3,
                dash: isTarget ? "dot" : "solid"
            }},
            marker: {{
                color: series.color,
                size: isTarget ? 9 : 6
            }},
            hovertemplate: "%{{x:.1f}}&deg; &rarr; %{{y:.1f}}&deg;" +
                "<extra>" + series.label + "</extra>"
        }});
    }}
    return traces;
}}

function curveLayout() {{
    return {{
        margin: {{ t: 16, r: 24, b: 48, l: 64 }},
        legend: {{ orientation: "h", y: -0.18 }},
        xaxis: {{
            title: {{ text: "Eingangswinkel [&deg;]" }},
            zeroline: false
        }},
        yaxis: {{
            title: {{ text: "Ausgangswinkel [&deg;]" }},
            zeroline: false
        }}
    }};
}}

Plotly.newPlot(
    "lever-plot",
    leverTraces(),
    leverLayout(),
    {{ responsive: true }}
);

Plotly.newPlot(
    "curve-plot",
    curveTraces(),
    curveLayout(),
    {{ responsive: true }}
);
</script>
</body>
</html>
"""
