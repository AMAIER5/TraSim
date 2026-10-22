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
    apply_parameter_overrides,
    is_angle_parameter,
    load_inputs,
    parameter_template,
    preview,
    optimized_values,
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
    )
    max_generations = st.number_input(
        "Maximale Generationen",
        min_value=10,
        max_value=5000,
        value=200,
        step=50,
    )
    children_count = st.number_input(
        "Kinder pro Generation",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )
    mutation_strength = st.slider(
        "Mutationsstärke",
        min_value=0.001,
        max_value=0.05,
        value=0.01,
        step=0.001,
        format="%.3f",
    )
    seed = st.number_input(
        "Zufallsseed",
        min_value=0,
        value=42,
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
    use_container_width=True,
    key="target_curve_editor",
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
                math.degrees(angle)
                for angle in inputs.input_angles
            ],
            "Ausgangswinkel (°)": [
                math.degrees(angle)
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
                math.degrees(angle)
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
    "Diese Tabelle wird automatisch aus der "
    "Mechanikdefinition erzeugt und aktualisiert "
    "sich nach jeder Änderung der Tabelle oben. "
    "Startwert und Grenzen können angepasst "
    "werden; gleiche Min-/Max-Grenzen fixieren "
    "einen Parameter."
)

mechanism_rows = [
    {
        "id": lever.id,
        "length_min": lever.length_min,
        "length_max": lever.length_max,
        "length_start": lever.length_start,
        "angle_min": math.degrees(
            lever.angle_min,
        ),
        "angle_max": math.degrees(
            lever.angle_max,
        ),
        "angle_start": math.degrees(
            lever.angle_start,
        ),
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
    for lever in inputs.definition.levers
]

edited_mechanism = st.data_editor(
    mechanism_rows,
    num_rows="dynamic",
    use_container_width=True,
    key="mechanism_editor",
    column_config={
        "id": st.column_config.NumberColumn(
            "id",
            help="Eindeutige Hebel-ID (erster Hebel = Antrieb)",
        ),
        "angle_min": st.column_config.NumberColumn(
            "angle_min (°)",
        ),
        "angle_max": st.column_config.NumberColumn(
            "angle_max (°)",
        ),
        "angle_start": st.column_config.NumberColumn(
            "angle_start (°)",
        ),
        "pivot_on": st.column_config.TextColumn(
            "pivot_on",
            help="lever_id@angle_deg (z. B. lever3@180), leer = fester Drehpunkt",
        ),
    },
)

try:
    inputs = with_mechanism_levers(
        inputs,
        edited_mechanism,
    )
except WorkflowError as error:
    st.error(str(error))
    st.stop()

template = parameter_template(inputs)

result = st.session_state.get("result")
optimized_map: dict[str, float] = {}
if result is not None:
    optimized_map = optimized_values(
        result.optimized_definition,
    )

# Pending single-parameter adoptions requested via
# the per-parameter buttons below.  They survive the
# rerun triggered by the button click.
if "adopted_params" not in st.session_state:
    st.session_state["adopted_params"] = set()

adopt_all = (
    st.session_state.get(
        "adopt_all_pressed",
        False,
    )
)
if adopt_all and result is not None:
    st.session_state[
        "adopted_params"
    ] = set(optimized_map)
if adopt_all:
    st.session_state["adopt_all_pressed"] = (
        False
    )

if optimized_map:
    st.caption(
        "✅ Optimierungsergebnis vorhanden — die "
        "Spalte „Optimiert“ zeigt die Werte des "
        "letzten Optimierungslaufs."
    )
    adopt_all_pressed = st.button(
        "Alle optimierten Werte als Startwert übernehmen",
        key="adopt_all_button",
    )
    if adopt_all_pressed:
        st.session_state["adopt_all_pressed"] = (
            True
        )
        st.rerun()

rows = []
for parameter in template.parameters:
    angle = is_angle_parameter(parameter)
    factor = (
        180.0 / math.pi if angle else 1.0
    )
    start_value = parameter.value
    if (
        parameter.name
        in st.session_state["adopted_params"]
        and parameter.name in optimized_map
    ):
        start_value = optimized_map[
            parameter.name
        ]
    rows.append(
        {
            "Parameter": parameter.name,
            "Typ": (
                "Winkel (°)"
                if angle
                else "Länge (mm)"
            ),
            "Minimum": (
                parameter.minimum * factor
            ),
            "Maximum": (
                parameter.maximum * factor
            ),
            "Startwert": (
                start_value * factor
            ),
            "Optimiert": (
                optimized_map.get(
                    parameter.name,
                )
                * factor
                if parameter.name
                in optimized_map
                else None
            ),
            "Optimieren": (
                not parameter.is_fixed
            ),
        },
    )

edited = st.data_editor(
    rows,
    num_rows="fixed",
    use_container_width=True,
    disabled=[
        "Parameter",
        "Typ",
        "Optimiert",
    ],
)

overrides: dict[
    str,
    tuple[float, float, float],
] = {}
for row, parameter in zip(
    edited,
    template.parameters,
):
    minimum = float(row["Minimum"])
    maximum = float(row["Maximum"])
    value = float(row["Startwert"])
    if not row["Optimieren"]:
        minimum = value
        maximum = value
    if maximum < minimum:
        st.error(
            f"Ungültiger Bereich für "
            f"{parameter.name}: "
            f"Maximum < Minimum."
        )
        st.stop()
    overrides[parameter.name] = (
        minimum,
        maximum,
        value,
    )

if optimized_map:
    st.subheader(
        "Optimierten Wert als Startwert übernehmen"
    )
    for parameter in template.parameters:
        if parameter.name not in optimized_map:
            continue
        column_name, column_button = (
            st.columns([0.7, 0.3])
        )
        with column_name:
            angle = is_angle_parameter(
                parameter,
            )
            unit = "°" if angle else " mm"
            st.write(
                f"**{parameter.name}** → "
                f"{optimized_map[parameter.name]:.4f}"
                f"{unit}"
            )
        with column_button:
            if st.button(
                "Übernehmen",
                key=(
                    f"adopt_{parameter.name}"
                ),
            ):
                st.session_state[
                    "adopted_params"
                ].add(parameter.name)
                st.rerun()

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
    if st.button("Vorschau aktualisieren"):
        try:
            preview_html = preview(inputs)
        except WorkflowError as error:
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
    )
    if start_pressed:
        settings = OptimizationSettings(
            population_size=int(population_size),
            children_count=int(children_count),
            mutation_strength=(
                float(mutation_strength)
            ),
            max_generations=int(max_generations),
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
        history = (
            result.fitness_history
        )
        if history:
            finite = [
                score
                if math.isfinite(score)
                else float("nan")
                for score in history
            ]
            status_text.empty()
            score_chart.line_chart(
                {
                    "Beste Fitness": finite,
                },
                x_label="Generation",
                y_label="Fitness",
            )

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
