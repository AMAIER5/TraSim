"""gui/app.py

Streamlit application for the TraSim workflow.

(1) The user uploads mechanism.csv and
    targetcurve.csv (as saved by Excel; German and
    international CSV conventions are detected).
(2) The user reviews and edits the optimization
    parameters derived from the mechanism definition
    and can preview the loaded mechanism and target
    curve in the usual HTML representation.
(3) The user starts the optimization and follows the
    live progress.
(4) The result is shown as the usual HTML document
    and offered as downloads (mechanism_optimized.csv
    and simulation_result.html).

Run with:

    streamlit run gui/app.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import streamlit as st

from analysis.target_curve import (
    MAX_WEIGHT,
)
from gui.mixed_brake_workflow import (
    MixedBrakeSettings,
    load_mixed_brake_inputs,
    mixed_brake_parameter_template,
    run_mixed_brake_optimization,
)
from gui.workflow import (
    LoadedInputs,
    OptimizationResult,
    OptimizationSettings,
    WorkflowError,
    adopt_single_optimized_values,
    apply_parameter_overrides,
    load_inputs,
    optimized_display_values,
    optimized_values,
    parameter_template,
    preview,
    run_optimization,
    with_mechanism_levers,
    with_target_points,
)

st.set_page_config(
    page_title="TraSim Optimierung",
    layout="wide",
)

MECHANISM_FILE = "mechanism.csv"
TARGET_FILE = "targetcurve.csv"


def _download_content() -> str:
    return ""


st.title("TraSim — Getriebeoptimierung")
st.caption(
    "Mechanismus und Zielkurve als CSV hochladen, "
    "Parameter anpassen, optimieren und das "
    "Ergebnis als HTML-Dokument speichern."
)

with st.sidebar:
    st.header("Dateien")
    mechanism_upload = st.file_uploader(
        "Mechanismus (mechanism.csv)",
        type=["csv"],
        key="mechanism",
    )
    target_upload = st.file_uploader(
        "Zielkurve (targetcurve.csv)",
        type=["csv"],
        key="target",
    )
    st.divider()
    st.header("Optimierung")
    population_size = st.number_input(
        "Populationsgröße",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help=(
            "Anzahl der Mechanismus-Kandidaten "
            "pro Generation. Größere "
            "Populationen durchsuchen den "
            "Parameterraum breiter, erhöhen "
            "aber die Rechenzeit pro Generation."
        ),
    )
    max_generations = st.number_input(
        "Maximale Generationen",
        min_value=10,
        max_value=5000,
        value=200,
        step=50,
        help=(
            "Obere Grenze der Optimierungs-"
            "generationen. Die Optimierung "
            "endet spätestens nach dieser "
            "Anzahl an Generationen."
        ),
    )
    children_count = st.number_input(
        "Kinder pro Generation",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help=(
            "Anzahl der Nachkommen, die pro "
            "Generation aus den ausgewählten "
            "Kandidaten erzeugt werden."
        ),
    )
    mutation_strength = st.slider(
        "Mutationsstärke",
        min_value=0.001,
        max_value=0.05,
        value=0.01,
        step=0.001,
        format="%.3f",
        help=(
            "Stärke der zufälligen "
            "Parameteränderung bei der "
            "Erzeugung von Nachkommen. Kleine "
            "Werte feinern Suchen, größere "
            "springen weiter im Parameterraum."
        ),
    )
    stagnation_limit = st.number_input(
        "Stagnationslimit (Generationen)",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        help=(
            "Anzahl aufeinanderfolgender "
            "Generationen ohne relevante "
            "Verbesserung, nach denen die "
            "Optimierung vorzeitig abgebrochen "
            "wird (Stoppgrund: "
            "stagnation_limit_reached)."
        ),
    )
    gradient_weight_input = st.slider(
        "Gradientengewicht",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        help=(
            "Gewichtung der Steigungsabweichung "
            "(Gradient) zusätzlich zum "
            "Stützpunktvergleich. Ein Wert "
            "von 0 vergleicht nur die "
            "Stützpunkte selbst; höhere "
            "Werte bestrafen Kurvenverläufe, "
            "die zwischen den Stützpunkten vom "
            "Zielverlauf abweichen."
        ),
    )
    workers_input = st.number_input(
        "Worker-Prozesse",
        min_value=0,
        max_value=64,
        value=0,
        step=1,
        help=(
            "Anzahl der Parallelprozesse für "
            "die Fitness-Bewertung. 0 = "
            "automatisch alle verfügbaren "
            "CPU-Kerne nutzen (Empfehlung). 1 = "
            "sequenzielle Berechnung in einem "
            "Prozess."
        ),
    )
    seed = st.number_input(
        "Zufallsseed",
        min_value=0,
        value=42,
        help=(
            "Startwert des Zufallsgenerators. "
            "Gleiche Seed-Werte liefern "
            "reproduzierbare Optimierungsläufe."
        ),
    )

if (
    mechanism_upload is None
    or target_upload is None
):
    st.info(
        "Bitte mechanism.csv und targetcurve.csv "
        "hochladen (Excel: „Speichern unter“ → CSV)."
    )
    st.stop()

try:
    inputs: LoadedInputs = load_inputs(
        mechanism_upload.getvalue().decode(
            "utf-8",
        ),
        target_upload.getvalue().decode(
            "utf-8",
        ),
    )
except WorkflowError as error:
    st.error(str(error))
    st.stop()

st.success(
    f"Mechanismus geladen: "
    f"{len(inputs.definition.levers)} Hebel · "
    f"Zielkurve: {inputs.point_count} Stützpunkte"
)

st.header("Zielkurve")
st.caption(
    "Die eingelesenen Stützpunkte der Zielkurve "
    "können hier direkt bearbeitet, ergänzt und "
    "gelöscht werden. Eingabewinkel müssen "
    "aufsteigend und eindeutig sein; Gewichte "
    "werden auf "
    f"{MAX_WEIGHT:.0f} begrenzt. Die Diagramme "
    "zeigen immer die aktuellen Werte."
)

target_rows = [
    {
        "Eingabewinkel (°)": math.degrees(
            angle,
        ),
        "Ausgangswinkel (°)": math.degrees(
            output,
        ),
        "Gewicht": weight,
    }
    for angle, output, weight in zip(
        inputs.input_angles,
        inputs.output_angles,
        inputs.target_curve.weights,
    )
]

edited_target = st.data_editor(
    target_rows,
    num_rows="dynamic",
    width="stretch",
    key="target_curve_editor",
    column_config={
        "Eingabewinkel (°)": (
            st.column_config.NumberColumn(
                "Eingabewinkel (°)",
                format="%.1f",
            )
        ),
        "Ausgangswinkel (°)": (
            st.column_config.NumberColumn(
                "Ausgangswinkel (°)",
                format="%.1f",
            )
        ),
        "Gewicht": (
            st.column_config.NumberColumn(
                "Gewicht",
                format="%.1f",
            )
        ),
    },
)

edited_target = sorted(
    edited_target,
    key=lambda row: row[
        "Eingabewinkel (°)"
    ],
)

try:
    inputs = with_target_points(
        inputs,
        [
            row["Eingabewinkel (°)"]
            for row in edited_target
        ],
        [
            row["Ausgangswinkel (°)"]
            for row in edited_target
        ],
        [
            row["Gewicht"]
            for row in edited_target
        ],
    )
except WorkflowError as error:
    st.error(str(error))
    st.stop()

curve_column, weight_column = st.columns(2)
with curve_column:
    st.subheader("Zielkurve")
    st.line_chart(
        {
            "Eingabewinkel (°)": [
                round(math.degrees(angle), 1)
                for angle in inputs.input_angles
            ],
            "Ausgangswinkel (°)": [
                round(math.degrees(angle), 1)
                for angle in inputs.output_angles
            ],
        },
        x="Eingabewinkel (°)",
        y="Ausgangswinkel (°)",
        x_label="Eingabewinkel (°)",
        y_label="Ausgangswinkel (°)",
    )
with weight_column:
    st.subheader("Gewichte")
    st.bar_chart(
        {
            "Eingabewinkel (°)": [
                round(math.degrees(angle), 1)
                for angle in inputs.input_angles
            ],
            "Gewicht": list(
                inputs.target_curve.weights,
            ),
        },
        x="Eingabewinkel (°)",
        y="Gewicht",
        x_label="Eingabewinkel (°)",
        y_label="Gewicht",
    )

st.header("Parameter des Startmechanismus")
st.caption(
    "Diese Tabelle ist die einzige Stelle, an der "
    "Startwerte, Grenzen und Freigabe der "
    "Optimierung gepflegt werden. Die Spalten "
    "„Optimiert“ zeigen das Ergebnis des "
    "letzten Optimierungslaufs und werden "
    "geleert, sobald der zugehörige Startwert "
    "geändert oder übernommen wird."
)

result = st.session_state.get("result")
optimized_map: dict[str, float] = {}
optimized_snapshot: dict[
    int,
    dict[str, float],
] = {}
if result is not None:
    optimized_map = optimized_values(
        result.optimized_definition,
    )
    optimized_snapshot = (
        st.session_state.get(
            "optimized_snapshot",
            {},
        )
    )

def _lever_rows(definition) -> list[dict]:
    return [
        {
            "id": lever.id,
            "length_min": lever.length_min,
            "length_max": lever.length_max,
            "length_start": lever.length_start,
            "length_optimize": True,
            "length_optimized": None,
            "length_adopt": False,
            "angle_min": math.degrees(
                lever.angle_min,
            ),
            "angle_max": math.degrees(
                lever.angle_max,
            ),
            "angle_start": math.degrees(
                lever.angle_start,
            ),
            "angle_optimize": True,
            "angle_optimized": None,
            "angle_adopt": False,
            "pivot_x": lever.pivot.x,
            "pivot_y": lever.pivot.y,
            "pivot_z": lever.pivot.z,
            "axis_x": lever.axis.x,
            "axis_y": lever.axis.y,
            "axis_z": lever.axis.z,
            "ref_x": (
                lever.reference_direction.x
                if lever.reference_direction
                else None
            ),
            "ref_y": (
                lever.reference_direction.y
                if lever.reference_direction
                else None
            ),
            "ref_z": (
                lever.reference_direction.z
                if lever.reference_direction
                else None
            ),
            "driver": lever.driver,
            "coupled": lever.coupled,
            "pivot_on": lever.pivot_on,
        }
        for lever in definition.levers
    ]


mechanism_bytes = (
    mechanism_upload.getvalue()
)
if (
    st.session_state.get(
        "mechanism_rows_source"
    )
    != mechanism_bytes
):
    st.session_state["mechanism_rows"] = (
        _lever_rows(inputs.definition)
    )
    st.session_state[
        "mechanism_rows_source"
    ] = mechanism_bytes
    st.session_state[
        "mechanism_rows_version"
    ] = (
        st.session_state.get(
            "mechanism_rows_version",
            0,
        )
        + 1
    )
    st.session_state.pop(
        "last_edited_rows",
        None,
    )


base_rows = st.session_state[
    "mechanism_rows"
]
reference_rows = st.session_state.get(
    "last_edited_rows",
    base_rows,
)
display_optimized = (
    optimized_display_values(
        reference_rows,
        optimized_map,
        optimized_snapshot,
    )
)

display_rows = copy.deepcopy(
    base_rows,
)
for row in display_rows:
    if row.get("id") is None:
        continue
    (
        row["length_optimized"],
        row["angle_optimized"],
    ) = display_optimized[
        int(row["id"])
    ]

edited_mechanism = st.data_editor(
    display_rows,
    num_rows="dynamic",
    width="stretch",
    key=(
        "mechanism_editor_"
        + str(
            st.session_state.get(
                "mechanism_rows_version",
                0,
            )
        )
    ),
    disabled=[
        "length_optimized",
        "angle_optimized",
    ],
    column_config={
        "id": st.column_config.NumberColumn(
            "id",
            help="Eindeutige Hebel-ID (erster Hebel = Antrieb)",
        ),
        "length_min": st.column_config.NumberColumn(
            "length_min",
            format="%.1f",
        ),
        "length_max": st.column_config.NumberColumn(
            "length_max",
            format="%.1f",
        ),
        "length_start": st.column_config.NumberColumn(
            "length_start",
            format="%.1f",
        ),
        "length_optimize": (
            st.column_config.CheckboxColumn(
                "Optimierung",
                default=True,
                help="Hebellänge für die Optimierung freigeben",
            )
        ),
        "length_optimized": (
            st.column_config.NumberColumn(
                "Optimiert",
                format="%.1f",
                help="Ergebnis des letzten Optimierungslaufs; wird geleert, sobald der Startwert geändert oder übernommen wird",
            )
        ),
        "length_adopt": (
            st.column_config.CheckboxColumn(
                "Übernehmen",
                default=False,
                help="Optimierten Wert als Startwert übernehmen",
            )
        ),
        "angle_min": st.column_config.NumberColumn(
            "angle_min (°)",
            format="%.1f",
        ),
        "angle_max": st.column_config.NumberColumn(
            "angle_max (°)",
            format="%.1f",
        ),
        "angle_start": st.column_config.NumberColumn(
            "angle_start (°)",
            format="%.1f",
            help=(
                "Startwinkel: Position des Hebels bei der "
                "mittleren Stützstelle der Zielkurve "
                "(n-te Position, n = Anzahl / 2, "
                "mathematisch gerundet)."
            ),
        ),
        "angle_optimize": (
            st.column_config.CheckboxColumn(
                "Optimierung",
                default=True,
                help="Hebelwinkel für die Optimierung freigeben",
            )
        ),
        "angle_optimized": (
            st.column_config.NumberColumn(
                "Optimiert",
                format="%.1f",
                help="Ergebnis des letzten Optimierungslaufs; wird geleert, sobald der Startwert geändert oder übernommen wird",
            )
        ),
        "angle_adopt": (
            st.column_config.CheckboxColumn(
                "Übernehmen",
                default=False,
                help="Optimierten Winkel als Startwert übernehmen",
            )
        ),
        "pivot_on": st.column_config.TextColumn(
            "pivot_on",
            help="lever_id (z. B. lever3), leer = fester Drehpunkt",
        ),
    },
)

st.session_state["last_edited_rows"] = (
    edited_mechanism
)

adopt_single = (
    adopt_single_optimized_values(
        edited_mechanism,
    )
)
if adopt_single:
    st.session_state[
        "mechanism_rows"
    ] = edited_mechanism
    st.session_state.pop(
        "last_edited_rows",
        None,
    )
    st.session_state[
        "mechanism_rows_version"
    ] = (
        st.session_state.get(
            "mechanism_rows_version",
            0,
        )
        + 1
    )
    st.rerun()

# Clear stale "Optimiert" cells as soon as the
# corresponding start value no longer matches
# the snapshot of the optimization run.
if optimized_display_values(
    edited_mechanism,
    optimized_map,
    optimized_snapshot,
) != display_optimized:
    st.rerun()

try:
    inputs = with_mechanism_levers(
        inputs,
        edited_mechanism,
    )
except WorkflowError as error:
    st.error(str(error))
    st.stop()

template = parameter_template(inputs)

if optimized_map and st.button(
    "Alle optimierten Werte als "
    "Startwerte übernehmen",
    key="adopt_all_button",
):
    for row in edited_mechanism:
        if row.get("id") is None:
            continue
        lever_id = int(row["id"])
        length_key = (
            f"lever.{lever_id}.length"
        )
        angle_key = (
            f"lever.{lever_id}.angle"
        )
        if length_key in optimized_map:
            row["length_start"] = (
                optimized_map[length_key]
            )
        if angle_key in optimized_map:
            row["angle_start"] = (
                math.degrees(
                    optimized_map[angle_key],
                )
            )
        row["length_adopt"] = False
        row["angle_adopt"] = False
    st.session_state[
        "mechanism_rows"
    ] = edited_mechanism
    st.session_state.pop(
        "last_edited_rows",
        None,
    )
    st.session_state[
        "mechanism_rows_version"
    ] = (
        st.session_state.get(
            "mechanism_rows_version",
            0,
        )
        + 1
    )
    st.rerun()

overrides: dict[
    str,
    tuple[float, float, float],
] = {}
for row in edited_mechanism:
    lever_id = int(row["id"])
    length_value = float(
        row["length_start"],
    )
    angle_value = float(
        row["angle_start"],
    )
    if row["length_optimize"]:
        length_min = float(
            row["length_min"],
        )
        length_max = float(
            row["length_max"],
        )
    else:
        length_min = length_max = (
            length_value
        )
    if row["angle_optimize"]:
        angle_min = float(
            row["angle_min"],
        )
        angle_max = float(
            row["angle_max"],
        )
    else:
        angle_min = angle_max = (
            angle_value
        )
    if length_max < length_min:
        st.error(
            f"Hebel {lever_id}: "
            f"length_max < length_min."
        )
        st.stop()
    if angle_max < angle_min:
        st.error(
            f"Hebel {lever_id}: "
            f"angle_max < angle_min."
        )
        st.stop()
    overrides[
        f"lever.{lever_id}.length"
    ] = (
        length_min,
        length_max,
        length_value,
    )
    overrides[
        f"lever.{lever_id}.angle"
    ] = (
        angle_min,
        angle_max,
        angle_value,
    )
    pivot_x = float(row["pivot_x"])
    pivot_y = float(row["pivot_y"])
    overrides[
        f"lever.{lever_id}.pivot.x"
    ] = (pivot_x, pivot_x, pivot_x)
    overrides[
        f"lever.{lever_id}.pivot.y"
    ] = (pivot_y, pivot_y, pivot_y)

try:
    active_template = (
        apply_parameter_overrides(
            template,
            overrides,
        )
    )
except ValueError as error:
    st.error(
        f"Ungültiger Parameterwert: {error}"
    )
    st.stop()

preview_tab, result_tab = st.tabs(
    ["Vorschau", "Ergebnis"]
)

with preview_tab:
    st.subheader("Vorschau")
    st.caption(
        "Simulation des Startmechanismus mit der "
        "geladenen Zielkurve."
    )
    refresh_preview = st.button(
        "Vorschau aktualisieren",
        key="refresh_preview_button",
    )
    if refresh_preview:
        try:
            preview_html = preview(
                inputs,
                active_template,
            )
        except WorkflowError as error:
            st.session_state.pop(
                "preview_html",
                None,
            )
            st.error(str(error))
        else:
            st.session_state["preview_html"] = (
                preview_html
            )
    preview_html = st.session_state.get(
        "preview_html",
    )
    if preview_html:
        st.components.v1.html(
            preview_html,
            height=900,
            scrolling=True,
        )
    else:
        st.info(
            "„Vorschau aktualisieren“ klicken, um "
            "die geladene Mechanik und die "
            "Zielkurve darzustellen."
        )

with result_tab:
    st.subheader("Ergebnis")
    start_pressed = st.button(
        "Optimierung starten",
        type="primary",
        key="start_optimization_button",
    )
    if start_pressed:
        settings = OptimizationSettings(
            population_size=int(population_size),
            children_count=int(children_count),
            mutation_strength=(
                float(mutation_strength)
            ),
            max_generations=int(max_generations),
            stagnation_limit=(
                int(stagnation_limit)
            ),
            gradient_weight=(
                float(gradient_weight_input)
            ),
            workers=int(workers_input),
            seed=int(seed),
        )
        progress_bar = st.progress(
            0.0,
            text="Optimierung läuft …",
        )
        status_text = st.empty()
        score_chart = st.empty()

        def report_progress(
            generation: int,
            best_score: float,
        ) -> None:
            fraction = min(
                generation
                / max(
                    1,
                    settings.max_generations,
                ),
                1.0,
            )
            progress_bar.progress(
                fraction,
                text=(
                    f"Generation {generation} · "
                    f"Fitness {best_score:.6f}"
                ),
            )
            status_text.write(
                f"**Generation:** {generation} · "
                f"**Beste Fitness:** "
                f"{best_score:.8f}"
            )

        try:
            result: OptimizationResult = (
                run_optimization(
                    inputs,
                    active_template,
                    settings,
                    progress=report_progress,
                )
            )
        except WorkflowError as error:
            st.error(str(error))
            st.stop()

        progress_bar.empty()
        st.session_state["result"] = result
        st.session_state[
            "optimized_snapshot"
        ] = {
            int(row["id"]): {
                "length": float(
                    row["length_start"],
                ),
                "angle": float(
                    row["angle_start"],
                ),
            }
            for row in edited_mechanism
            if row.get("id") is not None
        }
        history = (
            result.fitness_history
        )
        if history:
            status_text.empty()
            score_chart.line_chart(
                {
                    "Beste Fitness": [
                        score
                        if math.isfinite(
                            score
                        )
                        else float("nan")
                        for score in history
                    ],
                },
                x_label="Generation",
                y_label="Fitness",
            )
        st.rerun()

result = st.session_state.get("result")
if result is not None:
    with result_tab:
        st.write(
            f"**Stopgrund:** "
            f"{result.stop_reason} · "
            f"**Beste Fitness:** "
            f"{result.best_score:.8f} · "
            f"**Generationen:** "
            f"{result.generations}"
        )
        if result.fitness_history:
            st.line_chart(
                {
                    "Beste Fitness": (
                        list(
                            result.fitness_history,
                        )
                    ),
                },
                x_label="Generation",
                y_label="Fitness",
            )
        st.download_button(
            "Ergebnis-HTML herunterladen",
            data=result.html,
            file_name=(
                "simulation_result.html"
            ),
            mime="text/html",
        )
        st.download_button(
            "Optimierte Mechanik "
            "herunterladen (CSV)",
            data=result.mechanism_csv,
            file_name=(
                "mechanism_optimized.csv"
            ),
            mime="text/csv",
        )
        st.components.v1.html(
            result.html,
            height=900,
            scrolling=True,
        )


# ----------------------------------------------------------------------
# Mixer (mixed brake) mode
# ----------------------------------------------------------------------


def _mixed_brake_example_dir():
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(meipass) / "examples"
    return (
        Path(__file__).resolve().parent.parent
        / "examples"
    )


with st.expander(
    "Mischer-Modus (Mixed Brake, 5 Bremsstellungen)"
):
    st.caption(
        "Optimierung eines Mischerhebels "
        "(pivot_on) über 5 Bremsstellungen "
        "(0/25/50/75/100 % Bremsweg) mit "
        "interpolierten Zielkurven und weichem "
        "Minimum-Übertragungswinkel-Kriterium."
    )
    mechanism_upload_mixer = st.file_uploader(
        "Mechanismus (mixed_brake_mechanism.csv)",
        type=["csv"],
        key="mechanism_mixer",
    )
    unbraked_upload = st.file_uploader(
        "Zielkurve ungebremst",
        type=["csv"],
        key="target_unbraked",
    )
    braked_upload = st.file_uploader(
        "Zielkurve voll gebremst",
        type=["csv"],
        key="target_braked",
    )
    load_examples = st.button(
        "Beispiel-Dateien laden"
    )
    if (
        load_examples
        or "mixer_loaded" in st.session_state
    ):
        if load_examples:
            example_dir = _mixed_brake_example_dir()
            st.session_state["mixer_loaded"] = (
                example_dir
            )
        st.session_state.setdefault(
            "mixer_loaded",
            _mixed_brake_example_dir(),
        )
        example_dir = st.session_state[
            "mixer_loaded"
        ]
        mixer_file = (
            example_dir
            / "mixed_brake_mechanism.csv"
        ).read_text(encoding="utf-8")
        unbraked_file = (
            example_dir
            / "target_curve_unbraked.csv"
        ).read_text(encoding="utf-8")
        braked_file = (
            example_dir
            / "target_curve_braked.csv"
        ).read_text(encoding="utf-8")
    elif (
        mechanism_upload_mixer is not None
        and unbraked_upload is not None
        and braked_upload is not None
    ):
        mixer_file = mechanism_upload_mixer \
            .getvalue() \
            .decode("utf-8")
        unbraked_file = unbraked_upload \
            .getvalue() \
            .decode("utf-8")
        braked_file = braked_upload \
            .getvalue() \
            .decode("utf-8")
    else:
        mixer_file = None

    if mixer_file is None:
        st.info(
            "Beispiel-Dateien laden oder drei "
            "CSVs hochladen."
        )
    else:
        brake_positions = st.slider(
            "Bremsstellungen",
            min_value=3,
            max_value=5,
            value=5,
            step=1,
        )
        transmission_angle = st.slider(
            "Min. Übertragungswinkel (°)",
            min_value=0.0,
            max_value=45.0,
            value=20.0,
            step=1.0,
        )
        start_mixer = st.button(
            "Mischer-Optimierung starten",
            type="primary",
        )
        if start_mixer:
            try:
                mixer_inputs = (
                    load_mixed_brake_inputs(
                        mixer_file,
                        unbraked_file,
                        braked_file,
                    )
                )
            except WorkflowError as error:
                st.error(str(error))
                st.stop()
            template_mixer = (
                mixed_brake_parameter_template(
                    mixer_inputs,
                )
            )
            mixer_settings = MixedBrakeSettings(
                brake_positions=int(
                    brake_positions
                ),
                transmission_angle_deg=(
                    float(transmission_angle)
                ),
            )
            progress_mixer = st.progress(
                0.0,
                text=(
                    "Mischer-Optimierung läuft …"
                ),
            )

            def report_mixer(
                generation: int,
                best_score: float,
            ) -> None:
                progress_mixer.progress(
                    min(
                        generation
                        / mixer_settings.max_generations,
                        1.0,
                    ),
                    text=(
                        f"Generation {generation} · "
                        f"Fitness {best_score:.6f}"
                    ),
                )

            try:
                mixer_result = (
                    run_mixed_brake_optimization(
                        mixer_inputs,
                        template_mixer,
                        mixer_settings,
                        progress=report_mixer,
                    )
                )
            except WorkflowError as error:
                st.error(str(error))
                st.stop()
            progress_mixer.empty()
            st.write(
                f"**Stopgrund:** "
                f"{mixer_result.stop_reason} · "
                f"**Beste Fitness:** "
                f"{mixer_result.best_score:.8f} · "
                f"**Generationen:** "
                f"{mixer_result.generations} · "
                f"**Min. Übertragungswinkel:** "
                f"{mixer_result.min_transmission_angle_deg:.1f}°"
            )
            st.line_chart(
                {
                    "Beste Fitness": (
                        list(
                            mixer_result.fitness_history,
                        )
                    ),
                },
                x_label="Generation",
                y_label="Fitness",
            )
            st.components.v1.html(
                mixer_result.html,
                height=900,
                scrolling=True,
            )
            st.download_button(
                "Mischer-Ergebnis-HTML "
                "herunterladen",
                data=mixer_result.html,
                file_name=(
                    "mixed_brake_result.html"
                ),
                mime="text/html",
            )
