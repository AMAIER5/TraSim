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
from pathlib import Path

import streamlit as st

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


# ----------------------------------------------------------------------
# Mixer (mixed brake) mode
# ----------------------------------------------------------------------


def _mixed_brake_example_dir():
    return (
        Path(__file__).parent.parent / "examples"
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
        mixer_file = example_dir / (
            "mixed_brake_mechanism.csv"
        )
        unbraked_file = example_dir / (
            "target_curve_unbraked.csv"
        )
        braked_file = example_dir / (
            "target_curve_braked.csv"
        )
    elif (
        mechanism_upload_mixer is not None
        and unbraked_upload is not None
        and braked_upload is not None
    ):
        mixer_file = mechanism_upload_mixer
        unbraked_file = unbraked_upload
        braked_file = braked_upload
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
