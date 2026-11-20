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
import tempfile
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import (
    RerunException,
    StopException,
)

from analysis.target_curve import (
    MAX_WEIGHT,
    TargetCurve,
)
from gui.manual_workflow import (
    brake_lever,
    coupler_rod_stages,
    manual_transmission_curves,
)
from gui.debug_export import (
    debug_export_json,
    debug_result_info,
    parameters_dict,
    target_curve_dict,
)
from gui.mixed_brake_workflow import (
    MixedBrakeSettings,
    build_mixed_brake_state,
    mixed_brake_inputs_from_parts,
    mixed_brake_parameter_template,
    mixed_brake_preview,
    run_mixed_brake_optimization,
    single_brake_position_report,
)
from gui.pivot_shift_workflow import (
    PivotShiftSettings,
    pivot_pair_candidates,
    pivot_shift_inputs_from_parts,
    pivot_shift_parameter_template,
    run_pivot_shift_optimization,
)
from mechanism_io.csv_reader import CsvReader
from gui.workflow import (
    LoadedInputs,
    OptimizationResult,
    OptimizationSettings,
    OptimizationStopped,
    WorkflowError,
    adopt_single_optimized_values,
    apply_parameter_overrides,
    load_inputs,
    mechanism_csv_text,
    merge_editor_edits,
    optimized_display_values,
    optimized_values,
    parameter_template,
    preview,
    store_session_state_stop_tolerant,
    run_optimization,
    with_mechanism_levers,
    with_target_points,
)
from optimization.csv_parameter_factory import (
    DEFAULT_PIVOT_OFFSET,
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

def _csv_sniff_pivot_pair(
    data: bytes,
) -> bool:
    """Check whether the uploaded mechanism CSV contains a
    lever pair sharing a common pivot (pivot shift mode,
    the replacement of the modelled brake lever).

    Strings (``None`` while no file is uploaded) disable the
    check like the mixed mode sniff does."""
    try:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mechanism.csv"
            path.write_bytes(data)
            definition = CsvReader.read_mechanism(
                path,
            )
        return len(
            pivot_pair_candidates(definition)
        ) > 0
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def _csv_sniff_mixed_mode(
    data: bytes,
) -> bool:
    """Cheap check whether the uploaded mechanism CSV
    contains a lever with pivot_on (mixed brake
    mode), used to enable the braked target curve
    picker before the file is fully parsed
    (issue #89)."""
    try:
        text = data.decode(
            "utf-8",
            errors="replace",
        )
    except AttributeError:
        return False
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        return False
    for delimiter in (";", ","):
        cells = lines[0].split(delimiter)
        if "pivot_on" in cells:
            index = cells.index("pivot_on")
            return any(
                len(
                    line.split(delimiter)
                )
                > index
                and line.split(delimiter)[
                    index
                ].strip()
                for line in lines[1:]
            )
    return False


with st.sidebar:
    st.header("Dateien")
    mechanism_upload = st.file_uploader(
        "Mechanismus (mechanism.csv)",
        type=["csv"],
        key="mechanism",
    )
    target_upload = st.file_uploader(
        "Zielkurve ungebremst "
        "(targetcurve.csv)",
        type=["csv"],
        key="target",
    )
    mechanism_bytes = (
        mechanism_upload.getvalue()
        if mechanism_upload is not None
        else None
    )
    braked_upload = st.file_uploader(
        "Zielkurve voll gebremst "
        "(target_curve_braked.csv)",
        type=["csv"],
        key="target_braked",
        disabled=(
            not _csv_sniff_mixed_mode(
                mechanism_bytes,
            )
            and not _csv_sniff_pivot_pair(
                mechanism_bytes,
            )
        ),
        help=(
            "Wird erst aktiviert, wenn ein "
            "Zwei-Kurven-Mechanismus geladen wurde "
            "(Mischer-Modus: Hebel mit pivot_on; "
            "Pivot-Shift-Modus: Hebelpaar mit "
            "gemeinsamem Drehpunkt)."
        ),
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
    interpolated_steps = st.number_input(
        "Interpolierte Bremsstellungen",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
        help=(
            "Anzahl der interpolierten Zwischen-"
            "stellungen zwischen der ungebremsten "
            "und der voll gebremsten Zielkurve, "
            "auf die im Mischer-Modus optimiert "
            "wird. 0 = nur die beiden Grenzkurven "
            "(schnellste Berechnung); "
            "Standard 3 = Zwischenstellung alle "
            "25 % Bremsweg."
        ),
    )

def _mixed_brake_example_dir():
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(meipass) / "examples"
    return (
        Path(__file__).resolve().parent.parent
        / "examples"
    )


def _has_pivot_on(definition) -> bool:
    return any(
        lever.pivot_on is not None
        for lever in definition.levers
    )


def _mixed_brake_example_files() -> (
    tuple[str, str, str]
):
    example_dir = _mixed_brake_example_dir()
    return (
        (
            example_dir
            / "mixed_brake_mechanism.csv"
        ).read_text(encoding="utf-8"),
        (
            example_dir
            / "target_curve_unbraked.csv"
        ).read_text(encoding="utf-8"),
        (
            example_dir
            / "target_curve_braked.csv"
        ).read_text(encoding="utf-8"),
    )


def _two_curve_example_files(
    pivot: bool,
) -> tuple[str, str, str]:
    """Example files of the currently active mode: the
    brake-lever mixer example or the pivot-shift
    (brake-lever-less) example; both share the
    unbraked/braked target curve examples."""
    if not pivot:
        return _mixed_brake_example_files()
    example_dir = _mixed_brake_example_dir()
    return (
        (
            example_dir
            / "pivot_shift_mechanism.csv"
        ).read_text(encoding="utf-8"),
        (
            example_dir
            / "target_curve_unbraked.csv"
        ).read_text(encoding="utf-8"),
        (
            example_dir
            / "target_curve_braked.csv"
        ).read_text(encoding="utf-8"),
    )


def _target_curve_from_csv_text(
    text: str,
):
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as temp:
        path = _Path(temp) / "target_curve.csv"
        path.write_text(text, encoding="utf-8")
        return TargetCurve.from_csv_strict(path)


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


mixed_mode = _has_pivot_on(inputs.definition)
pivot_pairs = (
    pivot_pair_candidates(inputs.definition)
    if not mixed_mode
    else []
)
pivot_mode = bool(pivot_pairs)

mixed_inputs = None
pivot_inputs = None
if mixed_mode or pivot_mode:
    mode_label = (
        "Mischer-Modus"
        if mixed_mode
        else "Pivot-Shift-Modus"
    )
    with st.expander(
        f"{mode_label}: Beispiel-Dateien laden"
    ):
        st.caption(
            "Der geladene Mechanismus enthält einen "
            "Hebel mit pivot_on (Mischerhebel). "
            "Zusätzlich zur Zielkurve ungebremst wird "
            "eine zweite Zielkurve für die voll "
            "gebremste Stellung benötigt (Zielkurve "
            "voll gebremst); die "
            "Zielkurven der übrigen Bremsstellungen "
            "werden interpoliert."
            if mixed_mode
            else "Der geladene Mechanismus enthält ein "
            "Hebelpaar mit gemeinsamem Drehpunkt. "
            "Der gebremste Zustand entsteht durch "
            "Verschieben dieses Drehpunkts im "
            "definierten Raum; zusätzlich zur "
            "Zielkurve ungebremst wird eine zweite "
            "Zielkurve für die voll gebremste "
            "Stellung benötigt."
        )
        load_examples = st.button(
            "Beispiel-Dateien laden",
            key=(
                "mixed_brake_examples_button"
                if mixed_mode
                else "pivot_shift_examples_button"
            ),
        )
        if load_examples:
            (
                mechanism_example,
                unbraked_example,
                braked_example,
            ) = _two_curve_example_files(
                pivot=pivot_mode,
            )
            st.session_state[
                "mixed_brake_mechanism"
            ] = mechanism_example
            st.session_state[
                "mixed_brake_unbraked"
            ] = unbraked_example
            st.session_state[
                "mixed_brake_braked"
            ] = braked_example
            st.rerun()
        braked_text = None
        if braked_upload is not None:
            braked_text = (
                braked_upload.getvalue().decode(
                    "utf-8",
                )
            )
        elif (
            "mixed_brake_mechanism"
            in st.session_state
        ):
            braked_text = st.session_state[
                "mixed_brake_braked"
            ]
        if braked_text is None:
            st.info(
                "Bitte die gebremste Zielkurve "
                "hochladen oder die Beispiel-Dateien "
                f"laden, um den {mode_label} zu "
                "starten."
            )
            st.stop()
    if mixed_mode:
        try:
            mixed_inputs = (
                mixed_brake_inputs_from_parts(
                    inputs.definition,
                    inputs.target_curve,
                    _target_curve_from_csv_text(
                        braked_text
                    ),
                )
            )
        except WorkflowError as error:
            st.error(str(error))
            st.stop()
    else:
        shift_lever_options = [
            min(pair) for pair in pivot_pairs
        ]
        if len(shift_lever_options) > 1:
            shift_lever_id = st.selectbox(
                "Shift-Hebel des verschobenen "
                "Drehpunkts",
                options=shift_lever_options,
                key="pivot_shift_lever",
            )
        else:
            shift_lever_id = shift_lever_options[0]
        try:
            pivot_inputs = (
                pivot_shift_inputs_from_parts(
                    inputs.definition,
                    inputs.target_curve,
                    _target_curve_from_csv_text(
                        braked_text
                    ),
                    shift_lever_id=shift_lever_id,
                )
            )
        except WorkflowError as error:
            st.error(str(error))
            st.stop()

st.success(
    f"Mechanismus geladen: "
    f"{len(inputs.definition.levers)} Hebel · "
    f"Zielkurve: {inputs.point_count} Stüzpunkte"
    + (
        " · Mischer-Modus (pivot_on)"
        if mixed_mode
        else (
            " · Pivot-Shift-Modus (Hebelpaar-Drehpunkt)"
            if pivot_mode
            else ""
        )
    )
)
INPUT_ANGLE_COL = "Eing. (\u00b0)"
UNBRAKED_TARGET_COL = "Ziel ungeb. (\u00b0)"
UNBRAKED_WEIGHT_COL = "Gew. u"
BRAKED_TARGET_COL = "Ziel geb. (\u00b0)"
BRAKED_WEIGHT_COL = "Gew. b"
SINGLE_TARGET_COL = "Ziel (\u00b0)"
SINGLE_WEIGHT_COL = "Gew."

two_curve_mode = (
    mixed_inputs is not None
    or pivot_inputs is not None
)

braked_curve = None
if two_curve_mode:
    st.header(
        "Zielkurven ungebremst / voll gebremst"
    )
    st.caption(
        "Ungebremste und gebremste Zielkurve in "
        "einer Tabelle: je St\u00fctzwinkel ein "
        "Zielwert und Gewicht pro Kurve. Eingabe-"
        "winkel m\u00fcssen aufsteigend und "
        "eindeutig sein; Gewichte werden auf "
        f"{MAX_WEIGHT:.0f} begrenzt. Leere "
        "Zielwert-Zellen lassen den St\u00fctzwinkel "
        "f\u00fcr die jeweilige Kurve weg."
    )

    def _curve_rows(
        angles, outputs, weights,
    ):
        return {
            round(math.degrees(angle), 6): {
                "output": math.degrees(output),
                "weight": weight,
            }
            for angle, output, weight in zip(
                angles,
                outputs,
                weights,
            )
        }

    unbraked_points = _curve_rows(
        inputs.input_angles,
        inputs.output_angles,
        inputs.target_curve.weights,
    )
    braked_points = _curve_rows(
        (
            mixed_inputs.braked
            if mixed_inputs is not None
            else pivot_inputs.braked
        ).input_angles,
        (
            mixed_inputs.braked
            if mixed_inputs is not None
            else pivot_inputs.braked
        ).output_angles,
        (
            mixed_inputs.braked
            if mixed_inputs is not None
            else pivot_inputs.braked
        ).weights,
    )
    merged_rows = []
    for angle in sorted(
        set(unbraked_points)
        | set(braked_points),
    ):
        point = unbraked_points.get(angle)
        braked_point = braked_points.get(angle)
        merged_rows.append(
            {
                INPUT_ANGLE_COL: angle,
                UNBRAKED_TARGET_COL: (
                    point["output"]
                    if point is not None
                    else None
                ),
                UNBRAKED_WEIGHT_COL: (
                    point["weight"]
                    if point is not None
                    else None
                ),
                BRAKED_TARGET_COL: (
                    braked_point["output"]
                    if braked_point is not None
                    else None
                ),
                BRAKED_WEIGHT_COL: (
                    braked_point["weight"]
                    if braked_point is not None
                    else None
                ),
            }
        )

    edited_curves = st.data_editor(
        merged_rows,
        num_rows="dynamic",
        width="stretch",
        key="two_curve_editor",
        column_config={
            INPUT_ANGLE_COL: (
                st.column_config.NumberColumn(
                    INPUT_ANGLE_COL,
                    width="small",
                    format="%.1f",
                )
            ),
            UNBRAKED_TARGET_COL: (
                st.column_config.NumberColumn(
                    UNBRAKED_TARGET_COL,
                    width="small",
                    format="%.2f",
                )
            ),
            UNBRAKED_WEIGHT_COL: (
                st.column_config.NumberColumn(
                    UNBRAKED_WEIGHT_COL,
                    width="small",
                    format="%.1f",
                )
            ),
            BRAKED_TARGET_COL: (
                st.column_config.NumberColumn(
                    BRAKED_TARGET_COL,
                    width="small",
                    format="%.2f",
                )
            ),
            BRAKED_WEIGHT_COL: (
                st.column_config.NumberColumn(
                    BRAKED_WEIGHT_COL,
                    width="small",
                    format="%.1f",
                )
            ),
        },
    )

    def _split_curve(
        rows,
        target_column,
        weight_column,
    ):
        points = sorted(
            (
                row
                for row in rows
                if row.get(target_column)
                is not None
            ),
            key=lambda row: row[
                INPUT_ANGLE_COL
            ],
        )
        return (
            tuple(
                row[INPUT_ANGLE_COL]
                for row in points
            ),
            tuple(
                row[target_column]
                for row in points
            ),
            tuple(
                (
                    row[weight_column]
                    if row.get(weight_column)
                    is not None
                    else 1.0
                )
                for row in points
            ),
        )

    try:
        inputs = with_target_points(
            inputs,
            *_split_curve(
                edited_curves,
                UNBRAKED_TARGET_COL,
                UNBRAKED_WEIGHT_COL,
            ),
        )
    except WorkflowError as error:
        st.error(str(error))
        st.stop()
    try:
        (
            braked_angles,
            braked_outputs,
            braked_weights,
        ) = _split_curve(
            edited_curves,
            BRAKED_TARGET_COL,
            BRAKED_WEIGHT_COL,
        )
        braked_curve = (
            TargetCurve.from_points_strict(
                # _split_curve returns angles/outputs in
                # USER units (degrees); the strict curve
                # expects RADIANS (issues #135, #136).
                tuple(
                    math.radians(angle)
                    for angle in braked_angles
                ),
                tuple(
                    math.radians(angle)
                    for angle in braked_outputs
                ),
                braked_weights,
            )
        )
    except ValueError as error:
        st.error(
            f"Ung\u00fcltige gebremste "
            f"Zielkurve: {error}",
        )
        st.stop()
    if mixed_inputs is not None:
        try:
            mixed_inputs = (
                mixed_brake_inputs_from_parts(
                    inputs.definition,
                    inputs.target_curve,
                    braked_curve,
                )
            )
        except WorkflowError as error:
            st.error(str(error))
            st.stop()
    else:
        try:
            pivot_inputs = (
                pivot_shift_inputs_from_parts(
                    inputs.definition,
                    inputs.target_curve,
                    braked_curve,
                    shift_lever_id=(
                        pivot_inputs
                        .shift_lever_id
                    ),
                    input_angles=tuple(
                        inputs.input_angles,
                    ),
                )
            )
        except WorkflowError as error:
            st.error(str(error))
            st.stop()
else:
    st.header("Zielkurve")
    st.caption(
        "Die eingelesenen St\u00fctzpunkte der Zielkurve "
        "k\u00f6nnen hier direkt bearbeitet, erg\u00e4nzt und "
        "gel\u00f6scht werden. Eingabewinkel m\u00fcssen "
        "aufsteigend und eindeutig sein; Gewichte "
        "werden auf "
        f"{MAX_WEIGHT:.0f} begrenzt. Die Diagramme "
        "zeigen immer die aktuellen Werte."
    )
    target_rows = [
        {
            INPUT_ANGLE_COL: math.degrees(
                angle,
            ),
            SINGLE_TARGET_COL: math.degrees(
                output,
            ),
            SINGLE_WEIGHT_COL: weight,
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
            INPUT_ANGLE_COL: (
                st.column_config.NumberColumn(
                    INPUT_ANGLE_COL,
                    width="small",
                    format="%.1f",
                )
            ),
            SINGLE_TARGET_COL: (
                st.column_config.NumberColumn(
                    SINGLE_TARGET_COL,
                    width="small",
                    format="%.2f",
                )
            ),
            SINGLE_WEIGHT_COL: (
                st.column_config.NumberColumn(
                    SINGLE_WEIGHT_COL,
                    width="small",
                    format="%.1f",
                )
            ),
        },
    )
    edited_target = sorted(
        edited_target,
        key=lambda row: row[
            INPUT_ANGLE_COL
        ],
    )
    try:
        inputs = with_target_points(
            inputs,
            [
                row[INPUT_ANGLE_COL]
                for row in edited_target
            ],
            [
                row[SINGLE_TARGET_COL]
                for row in edited_target
            ],
            [
                row[SINGLE_WEIGHT_COL]
                for row in edited_target
            ],
        )
    except WorkflowError as error:
        st.error(str(error))
        st.stop()

def _braked_chart_curve():
    if mixed_inputs is not None:
        return mixed_inputs.braked
    if pivot_inputs is not None:
        return pivot_inputs.braked
    return None


curve_column, weight_column = st.columns(2)
with curve_column:
    st.subheader(
        "Zielkurve ungebremst"
        if mixed_mode
        else "Zielkurve"
    )
    chart_data = {
        "Eingabewinkel (°)": [
            round(math.degrees(angle), 1)
            for angle in inputs.input_angles
        ],
        "Ungebremst (°)": [
            round(math.degrees(angle), 1)
            for angle in inputs.output_angles
        ],
    }
    if (
        _braked_chart_curve() is not None
        and len(
            _braked_chart_curve().input_angles
        )
        == len(inputs.input_angles)
    ):
        chart_data[
            "Voll gebremst (°)"
        ] = [
            round(
                math.degrees(angle),
                1,
            )
            for angle in (
                _braked_chart_curve().output_angles
            )
        ]
    y_columns = ["Ungebremst (°)"]
    if "Voll gebremst (°)" in chart_data:
        y_columns.append(
            "Voll gebremst (°)"
        )
    st.line_chart(
        chart_data,
        x="Eingabewinkel (°)",
        y=y_columns,
        x_label="Eingabewinkel (°)",
        y_label="Ausgangswinkel (°)",
    )
with weight_column:
    st.subheader("Gewichte")
    weight_data = {
        "Eingabewinkel (°)": [
            round(math.degrees(angle), 1)
            for angle in inputs.input_angles
        ],
        "Ungebremst": list(
            inputs.target_curve.weights,
        ),
    }
    if (
        _braked_chart_curve() is not None
        and len(
            _braked_chart_curve().weights
        )
        == len(inputs.input_angles)
    ):
        weight_data[
            "Voll gebremst"
        ] = list(
            _braked_chart_curve().weights,
        )
    weight_y_columns = ["Ungebremst"]
    if "Voll gebremst" in weight_data:
        weight_y_columns.append(
            "Voll gebremst"
        )
    st.bar_chart(
        weight_data,
        x="Eingabewinkel (°)",
        y=weight_y_columns,
        x_label="Eingabewinkel (°)",
        y_label="Gewicht",
        stack=False,
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
mixed_result_state = st.session_state.get(
    "mixed_brake_result"
)
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
elif (
    mixed_result_state is not None
    and (
        mixed_result_state
        .optimized_definition
        is not None
    )
):
    optimized_map = optimized_values(
        mixed_result_state.optimized_definition,
    )
    optimized_snapshot = (
        st.session_state.get(
            "optimized_snapshot",
            {},
        )
    )

intermediate_best = st.session_state.get(
    "intermediate_best",
)
intermediate_generation: int | None = (
    None
)
if (
    not optimized_map
    and isinstance(
        intermediate_best,
        dict,
    )
):
    optimized_map = {
        str(name): float(value)
        for name, value in (
            intermediate_best.get(
                "values",
                {},
            )
        ).items()
        if value is not None
    }
    optimized_snapshot = (
        st.session_state.get(
            "optimized_snapshot",
            {},
        )
    )
    intermediate_generation = (
        intermediate_best.get(
            "generation",
        )
    )

_PIVOT_COLUMN_ORDER = (
    "pivot_x_min",
    "pivot_x_max",
    "pivot_x",
    "pivot_x_optimize",
    "pivot_x_optimized",
    "pivot_x_adopt",
    "pivot_y_min",
    "pivot_y_max",
    "pivot_y",
    "pivot_y_optimize",
    "pivot_y_optimized",
    "pivot_y_adopt",
    "pivot_z_min",
    "pivot_z_max",
    "pivot_z",
    "pivot_z_optimize",
    "pivot_z_optimized",
    "pivot_z_adopt",
)

_BASE_COLUMN_ORDER = (
    "id",
    "length_min",
    "length_max",
    "length",
    "length_optimize",
    "length_optimized",
    "length_adopt",
    "angle_min",
    "angle_max",
    "angle",
    "angle_optimize",
    "angle_optimized",
    "angle_adopt",
    "brake_angle",
    "brake_angle_optimize",
    "pivot_x",
    "pivot_y",
    "pivot_z",
    "axis_x",
    "axis_y",
    "axis_z",
    "ref_x",
    "ref_y",
    "ref_z",
    "driver",
    "coupled",
    "pivot_on",
)


def _mechanism_column_order(
    show_pivot_columns: bool,
) -> tuple[str, ...]:
    """Column order of the mechanism table.

    Follows the order requested in issue #92; the
    pivot optimization columns (min/max/optimize/
    optimized/adopt per pivot axis) are hidden by
    default and only shown when the user expands
    them, keeping the table readable (issue #83).
    """
    if not show_pivot_columns:
        return _BASE_COLUMN_ORDER
    expanded = (
        _BASE_COLUMN_ORDER[:13]
        + _PIVOT_COLUMN_ORDER
        + _BASE_COLUMN_ORDER[16:]
    )
    return expanded


def _brake_lever_ids(definition) -> set:
    # levers referenced via pivot_on = the adjusting
    # (brake) levers of a mixed mechanism
    return {
        lever.pivot_reference.lever_id
        for lever in definition.levers
        if lever.pivot_on is not None
    }


def _lever_rows(definition) -> list[dict]:
    brake_levers = _brake_lever_ids(definition)
    return [
        {
            "id": lever.id,
            "length_min": lever.length_min,
            "length_max": lever.length_max,
            "length": lever.length_start,
            "length_optimize": True,
            "length_optimized": None,
            "length_adopt": False,
            "angle_min": math.degrees(
                lever.angle_min,
            ),
            "angle_max": math.degrees(
                lever.angle_max,
            ),
            "angle": math.degrees(
                lever.angle_start,
            ),
            "angle_optimize": True,
            "angle_optimized": None,
            "angle_adopt": False,
            # Issue #127: the fully braked position of a brake
            # lever as its OWN attribute, separately editable
            # from the unbraked start value (angle) and from the
            # angle_max optimization bound.  Only adjusting
            # (brake) levers carry a value.
            "brake_angle": (
                math.degrees(
                    lever.resolved_brake_position,
                )
                if lever.id in brake_levers
                else None
            ),
            "brake_angle_optimize": True,
            "pivot_x_min": (
                lever.pivot_x_min
                if lever.pivot_x_min
                is not None
                else (
                    lever.pivot.x
                    - DEFAULT_PIVOT_OFFSET
                )
            ),
            "pivot_x_max": (
                lever.pivot_x_max
                if lever.pivot_x_max
                is not None
                else (
                    lever.pivot.x
                    + DEFAULT_PIVOT_OFFSET
                )
            ),
            "pivot_x": lever.pivot.x,
            "pivot_x_optimize": False,
            "pivot_x_optimized": None,
            "pivot_x_adopt": False,
            "pivot_y_min": (
                lever.pivot_y_min
                if lever.pivot_y_min
                is not None
                else (
                    lever.pivot.y
                    - DEFAULT_PIVOT_OFFSET
                )
            ),
            "pivot_y_max": (
                lever.pivot_y_max
                if lever.pivot_y_max
                is not None
                else (
                    lever.pivot.y
                    + DEFAULT_PIVOT_OFFSET
                )
            ),
            "pivot_y": lever.pivot.y,
            "pivot_y_optimize": False,
            "pivot_y_optimized": None,
            "pivot_y_adopt": False,
            "pivot_z_min": (
                lever.pivot_z_min
                if lever.pivot_z_min
                is not None
                else (
                    lever.pivot.z
                    - DEFAULT_PIVOT_OFFSET
                )
            ),
            "pivot_z_max": (
                lever.pivot_z_max
                if lever.pivot_z_max
                is not None
                else (
                    lever.pivot.z
                    + DEFAULT_PIVOT_OFFSET
                )
            ),
            "pivot_z": lever.pivot.z,
            "pivot_z_optimize": False,
            "pivot_z_optimized": None,
            "pivot_z_adopt": False,
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
    st.session_state.pop(
        "result",
        None,
    )
    st.session_state.pop(
        "mixed_brake_result",
        None,
    )
    st.session_state.pop(
        "optimized_snapshot",
        None,
    )
    st.session_state.pop(
        "pivot_shift_result",
        None,
    )
    st.session_state.pop(
        "intermediate_best",
        None,
    )
    st.session_state.pop(
        "preview_html",
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
        row["pivot_x_optimized"],
        row["pivot_y_optimized"],
        row["pivot_z_optimized"],
    ) = display_optimized[
        int(row["id"])
    ]

show_pivot_columns = st.toggle(
    "Drehpunkt-Optimierung anzeigen",
    value=False,
    help=(
        "Blendet die Spalten zur Freigabe und "
        "Übernahme der optimierten Drehpunkt-"
        "Positionen (pivot_x / pivot_y / "
        "pivot_z) ein. Standardmäßig sind sie "
        "ausgeblendet, um die Tabelle "
        "übersichtlich zu halten; die Drehpunkt-"
        "Optimierung ist standardmäßig "
        "deaktiviert."
    ),
)

mechanism_editor_key = (
    "mechanism_editor_"
    + str(
        st.session_state.get(
            "mechanism_rows_version",
            0,
        )
    )
)


def _persist_mechanism_edits():
    # Issue #126: the first input in the parameter table
    # must be durable at the moment the edit event is
    # processed, before the script runs.  Merging the
    # editor's diff into the source rows here guarantees
    # the value survives even when the rerun triggered by
    # the edit aborts early (validation error, st.rerun,
    # st.stop), so the first input is never withdrawn and
    # reset to the previous value.
    edits = st.session_state.get(
        mechanism_editor_key,
    )
    if not edits:
        return
    current_rows = st.session_state.get(
        "mechanism_rows",
    )
    if current_rows is None:
        return
    st.session_state["mechanism_rows"] = (
        merge_editor_edits(
            current_rows,
            edits,
        )
    )


edited_mechanism = st.data_editor(
    display_rows,
    num_rows="dynamic",
    width="stretch",
    column_order=(
        _mechanism_column_order(
            show_pivot_columns,
        )
    ),
    key=mechanism_editor_key,
    on_change=_persist_mechanism_edits,
    disabled=[
        "length_optimized",
        "angle_optimized",
    ],
    column_config={
        "id": st.column_config.NumberColumn(
            "id",
            width="small",help="Eindeutige Hebel-ID (erster Hebel = Antrieb)",
        ),
        "length_min": st.column_config.NumberColumn(
            "length_min",
            width="small",format="%.1f",
        ),
        "length_max": st.column_config.NumberColumn(
            "length_max",
            width="small",format="%.1f",
        ),
        "length": st.column_config.NumberColumn(
            "length",
            width="small",format="%.1f",
        ),
        "length_optimize": (
            st.column_config.CheckboxColumn(
                "Optimierung",
                width="small",default=True,
                help="Hebellänge für die Optimierung freigeben",
            )
        ),
        "length_optimized": (
            st.column_config.NumberColumn(
                "Optimiert",
                width="small",format="%.1f",
                help="Ergebnis des letzten Optimierungslaufs; wird geleert, sobald der Startwert geändert oder übernommen wird",
            )
        ),
        "length_adopt": (
            st.column_config.CheckboxColumn(
                "Übernehmen",
                width="small",default=False,
                help="Optimierten Wert als Startwert übernehmen",
            )
        ),
        "angle_min": st.column_config.NumberColumn(
            "angle_min (°)",
            width="small",format="%.1f",
        ),
        "angle_max": st.column_config.NumberColumn(
            "angle_max (°)",
            width="small",format="%.1f",
        ),
        "angle": st.column_config.NumberColumn(
            "angle (°)",
            width="small",format="%.1f",
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
                width="small",default=True,
                help="Hebelwinkel für die Optimierung freigeben",
            )
        ),
        "angle_optimized": (
            st.column_config.NumberColumn(
                "Optimiert",
                width="small",format="%.1f",
                help="Ergebnis des letzten Optimierungslaufs; wird geleert, sobald der Startwert geändert oder übernommen wird",
            )
        ),
        "angle_adopt": (
            st.column_config.CheckboxColumn(
                "Übernehmen",
                width="small",default=False,
                help="Optimierten Winkel als Startwert übernehmen",
            )
        ),
        "brake_angle": (
            st.column_config.NumberColumn(
                "Bremsw. (100 %)",
                width="small",format="%.1f",
                help=(
                    "Issue #127: eigene Anfangsposition des "
                    "Bremshebels im voll gebremsten Zustand, "
                    "unabhängig vom ungebremsten Startwinkel "
                    "(angle) und vom Optimierungs-Grenzwert "
                    "(angle_max)."
                ),
            )
        ),
        "brake_angle_optimize": (
            st.column_config.CheckboxColumn(
                "Optimierung",
                width="small",default=True,
                help="Voll gebremste Position für die Optimierung freigeben",
            )
        ),
        "pivot_x_min": st.column_config.NumberColumn(
            "pivot_x_min",
            width="small",format="%.1f",
        ),
        "pivot_x_max": st.column_config.NumberColumn(
            "pivot_x_max",
            width="small",format="%.1f",
        ),
        "pivot_x": (
            st.column_config.NumberColumn(
                "pivot_x",
                width="small",format="%.1f",
            )
        ),
        "pivot_x_optimize": (
            st.column_config.CheckboxColumn(
                "Optimierung",
                width="small",default=False,
                help="Drehpunkt-x für die Optimierung freigeben",
            )
        ),
        "pivot_x_optimized": (
            st.column_config.NumberColumn(
                "Optimiert",
                width="small",format="%.1f",
                help="Ergebnis des letzten Optimierungslaufs; wird geleert, sobald der Startwert geändert oder übernommen wird",
            )
        ),
        "pivot_x_adopt": (
            st.column_config.CheckboxColumn(
                "Übernehmen",
                width="small",default=False,
                help="Optimierte Drehpunkt-x-Position als Startwert übernehmen",
            )
        ),
        "pivot_y_min": st.column_config.NumberColumn(
            "pivot_y_min",
            width="small",format="%.1f",
        ),
        "pivot_y_max": st.column_config.NumberColumn(
            "pivot_y_max",
            width="small",format="%.1f",
        ),
        "pivot_y": (
            st.column_config.NumberColumn(
                "pivot_y",
                width="small",format="%.1f",
            )
        ),
        "pivot_y_optimize": (
            st.column_config.CheckboxColumn(
                "Optimierung",
                width="small",default=False,
                help="Drehpunkt-y für die Optimierung freigeben",
            )
        ),
        "pivot_y_optimized": (
            st.column_config.NumberColumn(
                "Optimiert",
                width="small",format="%.1f",
                help="Ergebnis des letzten Optimierungslaufs; wird geleert, sobald der Startwert geändert oder übernommen wird",
            )
        ),
        "pivot_y_adopt": (
            st.column_config.CheckboxColumn(
                "Übernehmen",
                width="small",default=False,
                help="Optimierte Drehpunkt-y-Position als Startwert übernehmen",
            )
        ),
        "pivot_z_min": st.column_config.NumberColumn(
            "pivot_z_min",
            width="small",format="%.1f",
        ),
        "pivot_z_max": st.column_config.NumberColumn(
            "pivot_z_max",
            width="small",format="%.1f",
        ),
        "pivot_z": (
            st.column_config.NumberColumn(
                "pivot_z",
                width="small",format="%.1f",
            )
        ),
        "pivot_z_optimize": (
            st.column_config.CheckboxColumn(
                "Optimierung",
                width="small",default=False,
                help="Drehpunkt-z für die Optimierung freigeben",
            )
        ),
        "pivot_z_optimized": (
            st.column_config.NumberColumn(
                "Optimiert",
                width="small",format="%.1f",
                help="Ergebnis des letzten Optimierungslaufs; wird geleert, sobald der Startwert geändert oder übernommen wird",
            )
        ),
        "pivot_z_adopt": (
            st.column_config.CheckboxColumn(
                "Übernehmen",
                width="small",default=False,
                help="Optimierte Drehpunkt-z-Position als Startwert übernehmen",
            )
        ),
        "pivot_on": st.column_config.TextColumn(
            "pivot_on",
            width="small",help="lever_id (z. B. lever3), leer = fester Drehpunkt",
        ),
    },
)

st.session_state["last_edited_rows"] = (
    edited_mechanism
)
st.session_state["mechanism_rows"] = (
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

if mixed_inputs is not None:
    template = mixed_brake_parameter_template(
        mixed_inputs
    )
elif pivot_inputs is not None:
    template = pivot_shift_parameter_template(
        pivot_inputs
    )
else:
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
        brake_angle_key = (
            f"lever.{lever_id}.brake_angle"
        )
        if length_key in optimized_map:
            row["length"] = (
                optimized_map[length_key]
            )
        if angle_key in optimized_map:
            row["angle"] = (
                math.degrees(
                    optimized_map[angle_key],
                )
            )
        if (
            brake_angle_key in optimized_map
        ):
            row["brake_angle"] = (
                math.degrees(
                    optimized_map[
                        brake_angle_key
                    ],
                )
            )
        pivot_x_key = (
            f"lever.{lever_id}.pivot.x"
        )
        pivot_y_key = (
            f"lever.{lever_id}.pivot.y"
        )
        pivot_z_key = (
            f"lever.{lever_id}.pivot.z"
        )
        if pivot_x_key in optimized_map:
            row["pivot_x"] = (
                optimized_map[pivot_x_key]
            )
        if pivot_y_key in optimized_map:
            row["pivot_y"] = (
                optimized_map[pivot_y_key]
            )
        if pivot_z_key in optimized_map:
            row["pivot_z"] = (
                optimized_map[pivot_z_key]
            )
        row["length_adopt"] = False
        row["angle_adopt"] = False
        row["pivot_x_adopt"] = False
        row["pivot_y_adopt"] = False
        row["pivot_z_adopt"] = False
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
        row["length"],
    )
    angle_value = float(
        row["angle"],
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
    # Issue #127: separate initial braked position of the
    # brake lever, bounded by the lever's angle segment
    # (or fixed to the start value when not released).
    brake_angle_raw = row.get("brake_angle")
    if brake_angle_raw is not None and (
        not isinstance(brake_angle_raw, str)
        or brake_angle_raw.strip()
    ):
        brake_angle_value = float(
            brake_angle_raw,
        )
        if angle_max < brake_angle_value:
            st.error(
                f"Hebel {lever_id}: "
                f"brake_angle > angle_max."
            )
            st.stop()
        if brake_angle_value < angle_min:
            st.error(
                f"Hebel {lever_id}: "
                f"brake_angle < angle_min."
            )
            st.stop()
        if row.get("brake_angle_optimize", True):
            brake_angle_min = angle_min
            brake_angle_max = angle_max
        else:
            brake_angle_min = (
                brake_angle_max
            ) = brake_angle_value
        overrides[
            f"lever.{lever_id}.brake_angle"
        ] = (
            brake_angle_min,
            brake_angle_max,
            brake_angle_value,
        )
    pivot_x_value = float(
        row["pivot_x"],
    )
    pivot_y_value = float(
        row["pivot_y"],
    )
    if row.get("pivot_x_optimize", False):
        pivot_x_min = float(row["pivot_x_min"])
        pivot_x_max = float(row["pivot_x_max"])
    else:
        pivot_x_min = pivot_x_max = (
            pivot_x_value
        )
    if row.get("pivot_y_optimize", False):
        pivot_y_min = float(row["pivot_y_min"])
        pivot_y_max = float(row["pivot_y_max"])
    else:
        pivot_y_min = pivot_y_max = (
            pivot_y_value
        )
    pivot_z_value = float(
        row["pivot_z"],
    )
    if row.get("pivot_z_optimize", False):
        pivot_z_min = float(row["pivot_z_min"])
        pivot_z_max = float(row["pivot_z_max"])
    else:
        pivot_z_min = pivot_z_max = (
            pivot_z_value
        )
    if pivot_x_max < pivot_x_min:
        st.error(
            f"Hebel {lever_id}: "
            f"pivot_x_max < pivot_x_min."
        )
        st.stop()
    if pivot_y_max < pivot_y_min:
        st.error(
            f"Hebel {lever_id}: "
            f"pivot_y_max < pivot_y_min."
        )
        st.stop()
    if pivot_z_max < pivot_z_min:
        st.error(
            f"Hebel {lever_id}: "
            f"pivot_z_max < pivot_z_min."
        )
        st.stop()
    overrides[
        f"lever.{lever_id}.pivot.x"
    ] = (
        pivot_x_min,
        pivot_x_max,
        pivot_x_value,
    )
    overrides[
        f"lever.{lever_id}.pivot.y"
    ] = (
        pivot_y_min,
        pivot_y_max,
        pivot_y_value,
    )
    overrides[
        f"lever.{lever_id}.pivot.z"
    ] = (
        pivot_z_min,
        pivot_z_max,
        pivot_z_value,
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

if mixed_inputs is not None:
    try:
        mixed_inputs = (
            mixed_brake_inputs_from_parts(
                inputs.definition,
                inputs.target_curve,
                mixed_inputs.braked,
            )
        )
        active_template = (
            apply_parameter_overrides(
                mixed_brake_parameter_template(
                    mixed_inputs
                ),
                overrides,
            )
        )
    except WorkflowError as error:
        st.error(str(error))
        st.stop()
elif pivot_inputs is not None:
    try:
        pivot_inputs = (
            pivot_shift_inputs_from_parts(
                inputs.definition,
                inputs.target_curve,
                pivot_inputs.braked,
                shift_lever_id=(
                    pivot_inputs.shift_lever_id
                ),
                input_angles=tuple(
                    inputs.input_angles,
                ),
            )
        )
        active_template = (
            apply_parameter_overrides(
                pivot_shift_parameter_template(
                    pivot_inputs
                ),
                overrides,
            )
        )
    except WorkflowError as error:
        st.error(str(error))
        st.stop()

if mixed_inputs is not None:
    preview_tab, manual_tab, result_tab = st.tabs(
        ["Vorschau", "Manuelle Optimierung", "Ergebnis"]
    )
else:
    preview_tab, result_tab = st.tabs(
        ["Vorschau", "Ergebnis"]
    )
    manual_tab = None

PREVIEW_BRAKE_KEY = (
    "mixed_preview_brake_position"
)


def _brake_positions(
    interpolated_steps: int,
) -> tuple[int, ...]:
    """Brake positions (%) of the mixed brake grid.

    ``interpolated_steps`` is the number of
    intermediate positions between the unbraked
    (0 %) and fully braked (100 %) curve; 0
    optimizes only the two limit curves
    (issue #90).
    """
    steps = max(0, int(interpolated_steps))
    if steps == 0:
        return (0, 100)
    return tuple(
        round(100 * index / (steps + 1))
        for index in range(steps + 2)
    )


def _position_count(
    interpolated_steps: int,
) -> int:
    return max(2, int(interpolated_steps) + 2)


BRAKE_POSITIONS = _brake_positions(
    interpolated_steps
)

with preview_tab:
    st.subheader("Vorschau")
    st.caption(
        "Simulation des Startmechanismus mit der "
        "geladenen Zielkurve."
        + (
            " Im Mischer-Modus wird der "
            "Mechanismus-Snapshot der gewählten "
            "Bremsstellung dargestellt."
            if mixed_inputs is not None
            else ""
        )
    )
    if mixed_inputs is not None:
        preview_brake = st.radio(
            "Bremsstellung",
            options=BRAKE_POSITIONS,
            format_func=(
                lambda value: f"{value} %"
            ),
            horizontal=True,
            key=PREVIEW_BRAKE_KEY,
        )
    refresh_preview = st.button(
        "Vorschau aktualisieren",
        key="refresh_preview_button",
    )
    if refresh_preview:
        try:
            if (
                mixed_inputs is not None
            ):
                preview_html = (
                    mixed_brake_preview(
                        mixed_inputs,
                        active_template,
                    )
                )
            else:
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
        if mixed_inputs is not None:
            try:
                preview_state = (
                    build_mixed_brake_state(
                        mixed_inputs,
                        active_template,
                        brake_fraction=(
                            preview_brake
                            / 100.0
                        ),
                        position_count=(
                            _position_count(
                                interpolated_steps
                            )
                        ),
                    )
                )
            except WorkflowError as error:
                st.error(str(error))
                preview_state = None
            if preview_state is not None:
                st.iframe(
                    single_brake_position_report(
                        (
                            "TraSim — Vorschau "
                            "(Startmechanismus, "
                            f"Bremsstellung "
                            f"{preview_brake} %)"
                        ),
                        preview_state,
                        show_actual=True,
                    ),
                    height=900,
                )
            else:
                st.iframe(
                    preview_html,
                    height=900,
                )
        else:
            st.iframe(
                preview_html,
                height=900,
            )
    else:
        st.info(
            "„Vorschau aktualisieren“ klicken, um "
            "die geladene Mechanik und die "
            "Zielkurve darzustellen."
        )

if manual_tab is not None:
    with manual_tab:
        st.subheader("Manuelle Optimierung")
        st.caption(
            "Koppelstangen sperren und einen Lagerpunkt "
            "oder den Drehwinkel des Bremshebels "
            "verstellen, dann die erzielte "
            "Übertragungskurve berechnen. Gesperrte "
            "Koppelstangen behalten ihre Länge; nicht "
            "gesperrte werden aus der aktuellen Geometrie "
            "neu berechnet."
        )
        rod_stages = coupler_rod_stages(inputs.definition)
        auto_lengths = {
            stage.driven_id: stage.length
            for stage in rod_stages
        }
        rod_source = (
            mechanism_bytes,
            st.session_state.get(
                "mechanism_rows_version",
                0,
            ),
        )
        if (
            st.session_state.get("manual_rod_source")
            != rod_source
        ):
            st.session_state["manual_rod_rows"] = [
                {
                    "Stufe": stage.stage_index + 1,
                    "Antrieb": stage.driver_id,
                    "Abtrieb": stage.driven_id,
                    "Länge (mm)": stage.length,
                    "Sperren": False,
                }
                for stage in rod_stages
            ]
            st.session_state["manual_rod_source"] = rod_source
            st.session_state["manual_rod_version"] = (
                st.session_state.get(
                    "manual_rod_version",
                    0,
                )
                + 1
            )
            st.session_state.pop("manual_curve", None)
        auto_signature = tuple(
            sorted(
                (driven_id, round(length, 6))
                for driven_id, length in auto_lengths.items()
            )
        )
        if (
            st.session_state.get(
                "manual_rod_auto_signature"
            )
            != auto_signature
        ):
            rod_rows = st.session_state.get(
                "manual_rod_rows"
            )
            if rod_rows is not None:
                for row in rod_rows:
                    driven_id = row.get("Abtrieb")
                    if (
                        not row.get("Sperren")
                        and driven_id in auto_lengths
                    ):
                        row["Länge (mm)"] = auto_lengths[
                            driven_id
                        ]
                st.session_state["manual_rod_version"] = (
                    st.session_state.get(
                        "manual_rod_version",
                        0,
                    )
                    + 1
                )
            st.session_state[
                "manual_rod_auto_signature"
            ] = auto_signature
        st.subheader("Koppelstangen")
        edited_rods = st.data_editor(
            st.session_state["manual_rod_rows"],
            num_rows="fixed",
            width="stretch",
            key=(
                "manual_rod_editor_"
                + str(
                    st.session_state.get(
                        "manual_rod_version",
                        0,
                    )
                )
            ),
            disabled=["Stufe", "Antrieb", "Abtrieb"],
            column_config={
                "Stufe": st.column_config.NumberColumn(
                    "Stufe",
                    width="small",format="%d",
                ),
                "Antrieb": st.column_config.NumberColumn(
                    "Antrieb (Hebel)",
                    width="small",format="%d",
                ),
                "Abtrieb": st.column_config.NumberColumn(
                    "Abtrieb (Hebel)",
                    width="small",format="%d",
                ),
                "Länge (mm)": st.column_config.NumberColumn(
                    "Länge (mm)",
                    width="small",format="%.2f",
                    help=(
                        "Gesperrt: Länge bleibt bei diesem "
                        "Wert, auch wenn Lagerpunkte oder "
                        "Hebellängen verändert werden. Nicht "
                        "gesperrt: Länge wird aus der "
                        "aktuellen Geometrie neu berechnet."
                    ),
                ),
                "Sperren": st.column_config.CheckboxColumn(
                    "Sperren",
                    width="small",default=False,
                    help=(
                        "Koppelstange von der Neuberechnung "
                        "ausnehmen (Länge bleibt fix)."
                    ),
                ),
            },
        )
        st.session_state["manual_rod_rows"] = edited_rods

        brake = brake_lever(inputs.definition)
        brake_min_deg = math.degrees(brake.angle_min)
        brake_max_deg = math.degrees(brake.angle_max)
        brake_start_deg = min(
            max(
                math.degrees(brake.angle_start),
                brake_min_deg,
            ),
            brake_max_deg,
        )
        brake_angle_deg = st.slider(
            f"Drehwinkel Bremshebel (Hebel {brake.id})",
            min_value=brake_min_deg,
            max_value=brake_max_deg,
            value=brake_start_deg,
            step=0.5,
            format="%.1f°",
            key="manual_brake_angle",
            help=(
                "Manueller Drehwinkel des Bremshebels; "
                "bestimmt die Bremsstellung der "
                "Übertragungskurve."
            ),
        )
        if st.button(
            "Übertragungskurve berechnen",
            type="primary",
            key="manual_curve_button",
        ):
            locked_rod_lengths = {
                int(row["Abtrieb"]): float(
                    row["Länge (mm)"]
                )
                for row in edited_rods
                if row.get("Sperren")
                and row.get("Abtrieb") is not None
            }
            try:
                st.session_state["manual_curve"] = (
                    manual_transmission_curves(
                        mixed_inputs,
                        inputs.definition,
                        rod_lengths=locked_rod_lengths,
                        brake_angle_deg=float(
                            brake_angle_deg
                        ),
                    )
                )
            except WorkflowError as error:
                st.session_state.pop(
                    "manual_curve",
                    None,
                )
                st.error(str(error))
        manual_curve = st.session_state.get("manual_curve")
        if manual_curve is not None:
            brake_pct = round(
                100 * manual_curve.brake_fraction
            )
            curve_chart_data = {
                "Eingabewinkel (°)": list(
                    manual_curve.input_angles_deg
                ),
                "Soll ungebremst (°)": list(
                    manual_curve.unbraked_deg
                ),
                "Soll voll gebremst (°)": list(
                    manual_curve.braked_deg
                ),
                f"Soll Bremse {brake_pct} % (°)": list(
                    manual_curve.target_deg
                ),
                f"Ist Bremse {brake_pct} % (°)": list(
                    manual_curve.actual_deg
                ),
            }
            curve_y_columns = [
                column
                for column in curve_chart_data
                if column != "Eingabewinkel (°)"
            ]
            st.line_chart(
                curve_chart_data,
                x="Eingabewinkel (°)",
                y=curve_y_columns,
                x_label="Eingabewinkel (°)",
                y_label="Ausgangswinkel (°)",
            )
            st.caption(
                "Drehwinkel Bremshebel: "
                f"{manual_curve.brake_angle_deg:.1f}° "
                f"(Bremsstellung {brake_pct} %)."
            )
        else:
            st.info(
                "„Übertragungskurve berechnen“ klicken, "
                "um die gebremste und ungebremste "
                "Solkurve sowie die erzielte "
                "Übertragungskurve darzustellen."
            )

with result_tab:
    st.subheader("Ergebnis")
    start_pressed = st.button(
        "Optimierung starten",
        type="primary",
        key="start_optimization_button",
    )
    if start_pressed:
        if mixed_inputs is not None:
            mixed_settings = MixedBrakeSettings(
                population_size=int(
                    population_size
                ),
                children_count=int(
                    children_count
                ),
                mutation_strength=(
                    float(mutation_strength)
                ),
                max_generations=int(
                    max_generations
                ),
                stagnation_limit=(
                    int(stagnation_limit)
                ),
                gradient_weight=(
                    float(
                        gradient_weight_input
                    )
                ),
                workers=int(
                    workers_input
                ),
                seed=int(seed),
            )
            progress_mixed = st.progress(
                0.0,
                text=(
                    "Mischer-Optimierung "
                    "läuft …"
                ),
            )
            st.button(
                "Optimierung stoppen",
                key="stop_mixed_button",
            )

            run_snapshot = {
                int(row["id"]): {
                    "length": float(
                        row["length"],
                    ),
                    "angle": float(
                        row["angle"],
                    ),
                    "pivot_x": float(
                        row["pivot_x"],
                    ),
                    "pivot_y": float(
                        row["pivot_y"],
                    ),
                    "pivot_z": float(
                        row["pivot_z"],
                    ),
                }
                for row in edited_mechanism
                if row.get("id") is not None
            }
            store_session_state_stop_tolerant(
                "optimized_snapshot",
                run_snapshot,
            )

            def report_mixed(
                generation: int,
                best_score: float,
            ) -> None:
                try:
                    progress_mixed.progress(
                        min(
                            generation
                            / max(
                                1,
                                (
                                    mixed_settings
                                    .max_generations
                                ),
                            ),
                            1.0,
                        ),
                        text=(
                            f"Generation "
                            f"{generation} · "
                            f"Fitness "
                            f"{best_score:.6f}"
                        ),
                    )
                except (
                    RerunException,
                    StopException,
                ) as error:
                    raise OptimizationStopped(
                        "Mischer-Optimierung vom "
                        "Benutzer gestoppt.",
                    ) from error

            def persist_best_mixed(
                generation: int,
                values: dict[str, float],
            ) -> None:
                store_session_state_stop_tolerant(
                    "intermediate_best",
                    {
                        "mode": "mixed",
                        "generation": (
                            generation
                        ),
                        "values": dict(
                            values,
                        ),
                    },
                )

            try:
                result = (
                    run_mixed_brake_optimization(
                        mixed_inputs,
                        active_template,
                        mixed_settings,
                        progress=report_mixed,
                        on_best=(
                            persist_best_mixed
                        ),
                    )
                )
            except WorkflowError as error:
                st.error(str(error))
                st.stop()
            store_session_state_stop_tolerant(
                "mixed_brake_result",
                result,
            )
            store_session_state_stop_tolerant(
                "result",
                None,
            )
            store_session_state_stop_tolerant(
                "intermediate_best",
                None,
            )
            try:
                progress_mixed.empty()
                st.rerun()
            except StopException:
                pass
        elif pivot_inputs is not None:
            pivot_settings = PivotShiftSettings(
                population_size=int(
                    population_size
                ),
                children_count=int(
                    children_count
                ),
                mutation_strength=(
                    float(mutation_strength)
                ),
                max_generations=int(
                    max_generations
                ),
                stagnation_limit=(
                    int(stagnation_limit)
                ),
                gradient_weight=(
                    float(
                        gradient_weight_input
                    )
                ),
                workers=int(
                    workers_input
                ),
                seed=int(seed),
            )
            progress_pivot = st.progress(
                0.0,
                text=(
                    "Pivot-Shift-Optimierung "
                    "läuft …"
                ),
            )
            st.button(
                "Optimierung stoppen",
                key="stop_pivot_button",
            )

            run_snapshot = {
                int(row["id"]): {
                    "length": float(
                        row["length"],
                    ),
                    "angle": float(
                        row["angle"],
                    ),
                    "pivot_x": float(
                        row["pivot_x"],
                    ),
                    "pivot_y": float(
                        row["pivot_y"],
                    ),
                    "pivot_z": float(
                        row["pivot_z"],
                    ),
                }
                for row in edited_mechanism
                if row.get("id") is not None
            }
            store_session_state_stop_tolerant(
                "optimized_snapshot",
                run_snapshot,
            )

            def report_pivot(
                generation: int,
                best_score: float,
            ) -> None:
                try:
                    progress_pivot.progress(
                        min(
                            generation
                            / max(
                                1,
                                (
                                    pivot_settings
                                    .max_generations
                                ),
                            ),
                            1.0,
                        ),
                        text=(
                            f"Generation "
                            f"{generation} · "
                            f"Fitness "
                            f"{best_score:.6f}"
                        ),
                    )
                except (
                    RerunException,
                    StopException,
                ) as error:
                    raise OptimizationStopped(
                        "Pivot-Shift-Optimierung "
                        "vom Benutzer gestoppt.",
                    ) from error

            def persist_best_pivot(
                generation: int,
                values: dict[str, float],
            ) -> None:
                store_session_state_stop_tolerant(
                    "intermediate_best",
                    {
                        "mode": "pivot",
                        "generation": (
                            generation
                        ),
                        "values": dict(
                            values,
                        ),
                    },
                )

            try:
                pivot_result = (
                    run_pivot_shift_optimization(
                        pivot_inputs,
                        active_template,
                        pivot_settings,
                        progress=report_pivot,
                        on_best=(
                            persist_best_pivot
                        ),
                    )
                )
            except WorkflowError as error:
                st.error(str(error))
                st.stop()
            store_session_state_stop_tolerant(
                "pivot_shift_result",
                pivot_result,
            )
            store_session_state_stop_tolerant(
                "result",
                None,
            )
            store_session_state_stop_tolerant(
                "mixed_brake_result",
                None,
            )
            store_session_state_stop_tolerant(
                "intermediate_best",
                None,
            )
            try:
                progress_pivot.empty()
                st.rerun()
            except StopException:
                pass
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
        st.button(
            "Optimierung stoppen",
            key="stop_optimization_button",
        )
        status_text = st.empty()
        score_chart = st.empty()

        run_snapshot = {
            int(row["id"]): {
                "length": float(
                    row["length"],
                ),
                "angle": float(
                    row["angle"],
                ),
                "pivot_x": float(
                    row["pivot_x"],
                ),
                "pivot_y": float(
                    row["pivot_y"],
                ),
                "pivot_z": float(
                    row["pivot_z"],
                ),
            }
            for row in edited_mechanism
            if row.get("id") is not None
        }
        store_session_state_stop_tolerant(
            "optimized_snapshot",
            run_snapshot,
        )

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
            try:
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
            except (
                RerunException,
                StopException,
            ) as error:
                raise OptimizationStopped(
                    "Optimierung vom "
                    "Benutzer gestoppt.",
                ) from error

        def persist_best(
            generation: int,
            values: dict[str, float],
        ) -> None:
            store_session_state_stop_tolerant(
                "intermediate_best",
                {
                    "mode": "standard",
                    "generation": (
                        generation
                    ),
                    "values": dict(
                        values,
                    ),
                },
            )

        try:
            result: OptimizationResult = (
                run_optimization(
                    inputs,
                    active_template,
                    settings,
                    progress=report_progress,
                    on_best=persist_best,
                )
            )
        except WorkflowError as error:
            st.error(str(error))
            st.stop()

        store_session_state_stop_tolerant(
            "result",
            result,
        )
        store_session_state_stop_tolerant(
            "mixed_brake_result",
            None,
        )
        store_session_state_stop_tolerant(
            "intermediate_best",
            None,
        )
        history = (
            result.fitness_history
        )
        try:
            progress_bar.empty()
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
        except StopException:
            pass

result = st.session_state.get("result")
mixed_result = st.session_state.get(
    "mixed_brake_result"
)
pivot_result = st.session_state.get(
    "pivot_shift_result"
)


def _debug_download_data() -> str:
    """JSON debug export of the current run: initial
    mechanism definition, optimized mechanism, target
    curve(s), all optimization settings, the manual settings
    (per-point weights, optimizable parameters with bounds)
    and the run outcome."""
    settings: dict = {
        "population_size": int(population_size),
        "children_count": int(children_count),
        "mutation_strength": float(mutation_strength),
        "max_generations": int(max_generations),
        "stagnation_limit": int(stagnation_limit),
        "gradient_weight": float(gradient_weight_input),
        "workers": int(workers_input),
        "seed": int(seed),
    }
    extra: dict = {
        "run_snapshot": (
            st.session_state.get(
                "optimized_snapshot"
            )
        ),
    }
    if mixed_inputs is not None:
        mode = "mixed_brake"
        result_obj = mixed_result
        settings["interpolated_steps"] = int(
            interpolated_steps
        )
        settings["brake_positions"] = _position_count(
            interpolated_steps
        )
        target_curves = [
            target_curve_dict(
                "unbraked",
                mixed_inputs.unbraked,
                inputs.input_angles,
            ),
            target_curve_dict(
                "braked",
                mixed_inputs.braked,
                inputs.input_angles,
            ),
        ]
        extra["brake_min_deg"] = (
            mixed_inputs.brake_min_deg
        )
        extra["brake_max_deg"] = (
            mixed_inputs.brake_max_deg
        )
        extra["motion_start_deg"] = (
            mixed_inputs.motion_start_deg
        )
        extra["motion_travel_deg"] = (
            mixed_inputs.motion_travel_deg
        )
        extra["motion_step_deg"] = (
            mixed_inputs.motion_step_deg
        )
        extra["result_brake_position_percent"] = (
            st.session_state.get(
                "mixed_result_brake_position"
            )
        )
        optimized_definition = (
            mixed_result.optimized_definition
            if mixed_result is not None
            else None
        )
        optimized_csv = (
            mixed_result.mechanism_csv
            if mixed_result is not None
            else ""
        )
    elif pivot_inputs is not None:
        mode = "pivot_shift"
        result_obj = pivot_result
        target_curves = [
            target_curve_dict(
                "unbraked",
                pivot_inputs.unbraked,
                inputs.input_angles,
            ),
            target_curve_dict(
                "braked",
                pivot_inputs.braked,
                inputs.input_angles,
            ),
        ]
        extra["shift_lever_id"] = (
            pivot_inputs.shift_lever_id
        )
        extra["shift_mm"] = (
            pivot_result.shift_mm
            if pivot_result is not None
            else None
        )
        optimized_definition = (
            pivot_result.optimized_definition
            if pivot_result is not None
            else None
        )
        optimized_csv = (
            pivot_result.mechanism_csv
            if pivot_result is not None
            else ""
        )
    else:
        mode = "single"
        result_obj = result
        target_curves = [
            target_curve_dict(
                "target_curve",
                inputs.target_curve,
                inputs.input_angles,
            ),
        ]
        optimized_definition = (
            result.optimized_definition
            if result is not None
            else None
        )
        optimized_csv = (
            result.mechanism_csv
            if result is not None
            else ""
        )
    return debug_export_json(
        mode=mode,
        initial_definition=inputs.definition,
        target_curves=target_curves,
        optimization_settings=settings,
        parameters=parameters_dict(active_template),
        result=(
            debug_result_info(result_obj)
            if result_obj is not None
            else {}
        ),
        optimized_definition=optimized_definition,
        initial_mechanism_csv=mechanism_csv_text(
            inputs.definition,
            inputs.mechanism_convention,
        ),
        optimized_mechanism_csv=optimized_csv,
        extra=extra,
    )


if mixed_result is not None:
    with result_tab:
        st.write(
            f"**Stopgrund:** "
            f"{mixed_result.stop_reason} · "
            f"**Beste Fitness:** "
            f"{mixed_result.best_score:.8f} · "
            f"**Generationen:** "
            f"{mixed_result.generations} · "
            f"**Min. Übertragungswinkel:** "
            f"{mixed_result.min_transmission_angle_deg:.1f}°"
        )
        if mixed_result.fitness_history:
            st.line_chart(
                {
                    "Beste Fitness": (
                        list(
                            mixed_result.fitness_history,
                        )
                    ),
                },
                x_label="Generation",
                y_label="Fitness",
            )
        result_brake = st.radio(
            "Bremsstellung",
            options=BRAKE_POSITIONS,
            format_func=(
                lambda value: f"{value} %"
            ),
            horizontal=True,
            key="mixed_result_brake_position",
        )
        try:
            if (
                mixed_result.optimized_definition
                is not None
            ):
                result_inputs = (
                    mixed_brake_inputs_from_parts(
                        (
                            mixed_result
                            .optimized_definition
                        ),
                        mixed_inputs.unbraked,
                        mixed_inputs.braked,
                    )
                )
                result_template = (
                    mixed_brake_parameter_template(
                        result_inputs
                    )
                )
                result_state = (
                    build_mixed_brake_state(
                        result_inputs,
                        result_template,
                        brake_fraction=(
                            result_brake
                            / 100.0
                        ),
                        position_count=(
                            _position_count(
                                interpolated_steps
                            )
                        ),
                    )
                )
            else:
                result_state = None
        except WorkflowError as error:
            st.error(str(error))
            result_state = None
        st.download_button(
            "Ergebnis-HTML herunterladen",
            data=mixed_result.html,
            file_name=(
                "mixed_brake_result.html"
            ),
            mime="text/html",
        )
        st.download_button(
            "Debug-Datei herunterladen (JSON)",
            data=_debug_download_data(),
            file_name="debug_export.json",
            mime="application/json",
        )
        if mixed_result.mechanism_csv:
            st.download_button(
                "Optimierte Mechanik "
                "herunterladen (CSV)",
                data=mixed_result.mechanism_csv,
                file_name=(
                    "mechanism_optimized.csv"
                ),
                mime="text/csv",
            )
        if result_state is not None:
            st.iframe(
                single_brake_position_report(
                    (
                        "TraSim — Ergebnis "
                        "(optimierte Geometrie, "
                        f"Bremsstellung "
                        f"{result_brake} %)"
                    ),
                    result_state,
                    show_actual=True,
                ),
                height=900,
            )
        else:
            # The selected brake position cannot be built
            # (issue #113): fall back to the overall report,
            # which still shows the target curves of every
            # brake position, instead of showing no curve.
            st.iframe(
                mixed_result.html,
                height=900,
            )
elif pivot_result is not None:
    with result_tab:
        st.write(
            f"**Stopgrund:** "
            f"{pivot_result.stop_reason} · "
            f"**Beste Fitness:** "
            f"{pivot_result.best_score:.8f} · "
            f"**Generationen:** "
            f"{pivot_result.generations} · "
            f"**Min. Übertragungswinkel:** "
            f"{pivot_result.min_transmission_angle_deg:.1f}° "
            f"· **Verschiebung:** "
            f"({pivot_result.shift_mm[0]:.2f}, "
            f"{pivot_result.shift_mm[1]:.2f}) mm"
        )
        if pivot_result.fitness_history:
            st.line_chart(
                {
                    "Beste Fitness": (
                        list(
                            pivot_result
                            .fitness_history,
                        )
                    ),
                },
                x_label="Generation",
                y_label="Fitness",
            )
        chart_left, chart_right = st.columns(2)

        def _result_chart_rows(
            target_deg,
            actual_deg,
        ):
            actual = list(actual_deg)
            return [
                {
                    "Eing. (°)": angle,
                    "Ziel (°)": target,
                    "Ist (°)": (
                        actual[index]
                        if index < len(actual)
                        else None
                    ),
                }
                for index, (
                    angle,
                    target,
                ) in enumerate(
                    zip(
                        pivot_result
                        .input_angles_deg,
                        target_deg,
                    )
                )
            ]

        target_actual_columns = [
            "Ziel (°)",
            "Ist (°)",
        ]
        with chart_left:
            st.subheader("Ungebremst")
            st.line_chart(
                _result_chart_rows(
                    pivot_result
                    .unbraked_target_deg,
                    pivot_result
                    .unbraked_actual_deg,
                ),
                x="Eing. (°)",
                y=target_actual_columns,
                x_label="Eing. (°)",
                y_label="Ausgang (°)",
            )
        with chart_right:
            st.subheader("Voll gebremst")
            st.line_chart(
                _result_chart_rows(
                    pivot_result
                    .braked_target_deg,
                    pivot_result
                    .braked_actual_deg,
                ),
                x="Eing. (°)",
                y=target_actual_columns,
                x_label="Eing. (°)",
                y_label="Ausgang (°)",
            )
        if not pivot_result.braked_buildable:
            st.warning(
                "Die gebremste Stellung des besten "
                "Kandidaten konnte nicht aufgebaut "
                "werden; die Verschiebung wurde "
                "reduziert und es gilt der "
                "reduzierte Zustand."
            )
        st.download_button(
            "Debug-Datei herunterladen (JSON)",
            data=_debug_download_data(),
            file_name="debug_export.json",
            mime="application/json",
        )
        if pivot_result.mechanism_csv:
            st.download_button(
                "Optimierte Mechanik "
                "herunterladen (CSV)",
                data=(pivot_result.mechanism_csv),
                file_name=(
                    "mechanism_optimized.csv"
                ),
                mime="text/csv",
            )
elif result is not None:
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
            "Debug-Datei herunterladen (JSON)",
            data=_debug_download_data(),
            file_name="debug_export.json",
            mime="application/json",
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
        st.iframe(
            result.html,
            height=900,
        )
elif intermediate_generation is not None:
    with result_tab:
        st.info(
            "Die Optimierung wurde vom Benutzer "
            "gestoppt; die Spalten „Optimiert“ "
            "zeigen den bis dahin besten "
            "Zwischenstand (Generation "
            f"{intermediate_generation}). "
            "Die Werte lassen sich über "
            "„Übernehmen“ als Startwerte "
            "übernehmen und dann erneut "
            "optimieren."
        )
