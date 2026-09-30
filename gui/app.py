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

import streamlit as st

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
    run_optimization,
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

template = parameter_template(inputs)

st.header("Optimierungsparameter")
st.caption(
    "Längen in mm, Winkel in Grad. Startwert und "
    "Grenzen können angepasst werden; gleiche "
    "Min-/Max-Grenzen fixieren einen Parameter."
)

rows = []
for parameter in template.parameters:
    angle = is_angle_parameter(parameter)
    factor = (
        180.0 / math.pi if angle else 1.0
    )
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
                parameter.value * factor
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
    disabled=["Parameter", "Typ"],
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
