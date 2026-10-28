"""gui/workflow.py

End-to-end workflow for interactive use:

    CSV inputs -> mechanism + target curve
        -> parameter template (editable)
        -> evolutionary optimization
        -> optimized mechanism CSV + HTML report

The module is independent of any particular UI
framework.  ``gui/app.py`` (Streamlit) is a thin layer
on top of it, and the unit tests exercise the workflow
directly.

User I/O angles are in DEGREES, internal calculations
use RADIANS.

Start angle semantics: the start angle of every lever
(its angle parameter, initially ``angle_start`` from
the mechanism CSV) describes the position the lever
occupies at the MIDDLE support position of the target
curve.  The middle position is the n-th support point
with n = point_count / 2, mathematically rounded half
up (seven support points -> fourth position).  The
target curve's input angles are applied as angular
offsets relative to that middle support position: the
drive lever sits at its start angle at the middle
support point, and every other support point keeps
its angular distance to the middle.
"""

from __future__ import annotations

import csv
import math
import multiprocessing
import random
import tempfile
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from functools import partial
from dataclasses import (
    dataclass,
    field,
    replace,
)
from pathlib import Path

from analysis.curve_fitness import (
    CurveFitness,
    GRADIENT_WEIGHT,
)
from analysis.curve_plotter import (
    CurvePlotter,
    mechanism_states_from_results,
)
from analysis.target_curve import (
    TargetCurve,
    DiscreteTargetCurve,
)
from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)
from mechanism_io.csv_convention import (
    GERMAN,
    CsvConvention,
    detect_convention,
    open_reader,
)
from mechanism_io.csv_reader import (
    CsvReader,
)
from mechanism_io.csv_writer import (
    CsvWriter,
)
from core.point3d import (
    Point3D,
)
from core.vector3d import (
    Vector3D,
)
from model.lever_definition import (
    LeverDefinition,
    parse_pivot_reference,
)
from model.mechanism_definition import (
    MechanismDefinition,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)
from optimization.parameter import (
    Parameter,
)
from optimization.parameter_mutation import (
    ParameterMutation,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import (
    Reproduction,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.point_motion import (
    PointMotion,
)
from simulation.stage_simulator import (
    StageSimulator,
)


class WorkflowError(Exception):
    """User-facing error in the optimization workflow."""


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """Settings of the evolutionary optimizer."""

    population_size: int = 100
    children_count: int = 100
    selection_count: int = 30
    mutation_strength: float = 0.01
    max_generations: int = 200
    stagnation_limit: int = 100
    stagnation_tolerance: float = 1e-8
    target_fitness: float = 0.01
    workers: int = 0
    gradient_weight: float = GRADIENT_WEIGHT
    seed: int = 42


@dataclass(frozen=True, slots=True)
class LoadedInputs:
    """Parsed and validated user inputs."""

    mechanism_convention: CsvConvention
    target_convention: CsvConvention
    output_convention: CsvConvention
    definition: MechanismDefinition
    target_curve: DiscreteTargetCurve
    input_angles: tuple[float, ...]
    output_angles: tuple[float, ...]
    point_count: int


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Outcome of a completed optimization run."""

    best_score: float
    stop_reason: str
    generations: int
    optimized_definition: MechanismDefinition
    mechanism_csv: str
    html: str
    fitness_history: tuple[float, ...] = field(
        default=(),
    )


def middle_position_index(
    point_count: int,
) -> int:
    """
    One-based index of the middle support position of a
    target curve with ``point_count`` positions.

    The middle position is the n-th position with
    n = point_count / 2, mathematically rounded
    (round half up).  Seven positions therefore give the
    fourth position, six positions the third.
    """

    return int(
        math.floor(point_count / 2 + 0.5)
    )


def middle_input_angle(
    inputs: LoadedInputs,
) -> float:
    """
    Input angle (radians) of the middle support position
    of the target curve.
    """

    index = middle_position_index(
        inputs.point_count,
    ) - 1
    return inputs.input_angles[index]


def middle_output_angle(
    inputs: LoadedInputs,
) -> float:
    """
    Output angle (radians) of the target curve at the
    middle support position.
    """

    index = middle_position_index(
        inputs.point_count,
    ) - 1
    return inputs.output_angles[index]


def anchored_input_angles(
    inputs: LoadedInputs,
    drive_angle: float,
) -> tuple[float, ...]:
    """
    Support-point input angles (radians) shifted so that
    the drive lever sits at ``drive_angle`` (its start
    angle) at the middle support position.

    Every support point keeps its angular distance to
    the middle position; only the origin of the drive
    lever's angle definition moves.
    """

    middle = middle_input_angle(inputs)
    return tuple(
        angle - middle + drive_angle
        for angle in inputs.input_angles
    )


def anchored_output_angles(
    inputs: LoadedInputs,
    output_angle: float,
) -> tuple[float, ...]:
    """
    Support-point output angles (radians) shifted so
    that the output lever sits at ``output_angle`` (its
    start angle) at the middle support position.
    """

    middle = middle_output_angle(inputs)
    return tuple(
        angle - middle + output_angle
        for angle in inputs.output_angles
    )


def _drive_lever_id(
    definition: MechanismDefinition,
) -> int:
    """
    Lever id of the drive lever (input lever of the first
    stage).
    """

    for lever in definition.levers:
        if lever.driver is not None:
            return lever.driver
    return definition.levers[0].id


def _output_lever_id(
    definition: MechanismDefinition,
) -> int:
    """
    Lever id of the output lever of the last stage.
    """

    output_id = None
    for lever in definition.levers:
        if lever.driver is not None:
            output_id = lever.id
    if output_id is None:
        return definition.levers[-1].id
    return output_id


def _anchored_motion(
    inputs: LoadedInputs,
    parameters: ParameterSet,
) -> PointMotion:
    """
    Motion over the target-curve support points anchored
    at the current start angle of the drive lever.
    """

    drive_id = _drive_lever_id(
        inputs.definition,
    )
    drive_angle = parameters.values().get(
        f"lever.{drive_id}.angle",
    )
    if drive_angle is None:
        drive_angle = (
            inputs.definition.get_lever(
                drive_id,
            ).angle_start
        )
    return PointMotion(
        angles=anchored_input_angles(
            inputs,
            drive_angle,
        ),
    )


def _anchored_simulator(
    inputs: LoadedInputs,
    parameters: ParameterSet,
) -> MechanismSimulator:
    """
    Simulator over the support points anchored at the
    drive lever's start angle (see
    ``anchored_input_angles``).
    """

    return MechanismSimulator(
        motion=_anchored_motion(
            inputs,
            parameters,
        ),
        stage_simulator=StageSimulator(),
        stage_limit=None,
    )


def _text_to_temp_file(
    directory: Path,
    name: str,
    text: str,
) -> Path:
    path = directory / name
    path.write_text(
        text,
        encoding="utf-8",
    )
    return path


def _read_target_points(
    path: Path,
    convention: CsvConvention,
) -> tuple[
    list[float],
    list[float],
    list[float],
]:
    input_angles: list[float] = []
    output_angles: list[float] = []
    weights: list[float] = []
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = open_reader(
            file,
            convention=convention,
        )
        has_weights = (
            reader.fieldnames is not None
            and "weight" in reader.fieldnames
        )
        for row in reader:
            input_angles.append(
                math.radians(
                    convention.parse_float(
                        row["input_angle"],
                    ),
                ),
            )
            output_angles.append(
                math.radians(
                    convention.parse_float(
                        row["output_angle"],
                    ),
                ),
            )
            if has_weights:
                weight_raw = row["weight"]
                weights.append(
                    convention.parse_float(
                        weight_raw,
                    )
                    if weight_raw
                    else 1.0
                )
            else:
                weights.append(1.0)
    if len(input_angles) < 2:
        raise WorkflowError(
            "Die Zielkurve benötigt mindestens "
            "zwei Stützpunkte.",
        )
    return input_angles, output_angles, weights


def load_inputs(
    mechanism_csv: str,
    target_csv: str,
) -> LoadedInputs:
    """Parse and validate mechanism and target curve CSV
    content (not paths).

    Raises :class:`WorkflowError` with a user-facing
    message when a file cannot be parsed.
    """

    try:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            mechanism_path = _text_to_temp_file(
                directory,
                "mechanism.csv",
                mechanism_csv,
            )
            target_path = _text_to_temp_file(
                directory,
                "target_curve.csv",
                target_csv,
            )
            mechanism_convention = (
                detect_convention(
                    mechanism_path,
                )
            )
            target_convention = (
                detect_convention(
                    target_path,
                )
            )
            definition = CsvReader.read_mechanism(
                mechanism_path,
            )
            input_deg, output_deg, weights = (
                _read_target_points(
                    target_path,
                    target_convention,
                )
            )
            if not definition.levers:
                raise WorkflowError(
                    "Die Mechanik-Datei enthält "
                    "keine Hebel. Bitte das "
                    "CSV-Format prüfen "
                    "(Spalten wie id, length_min, "
                    "…, driver, coupled).",
                )
            target_curve = (
                TargetCurve.from_points_strict(
                    tuple(input_deg),
                    tuple(output_deg),
                    tuple(weights),
                )
            )
    except WorkflowError:
        raise
    except Exception as error:
        raise WorkflowError(
            f"Eingabedatei konnte nicht gelesen "
            f"werden: {error}",
        ) from error

    output_convention = (
        GERMAN
        if (
            mechanism_convention.is_german
            or target_convention.is_german
        )
        else mechanism_convention
    )

    return LoadedInputs(
        mechanism_convention=(
            mechanism_convention
        ),
        target_convention=target_convention,
        output_convention=output_convention,
        definition=definition,
        target_curve=target_curve,
        input_angles=tuple(input_deg),
        output_angles=tuple(output_deg),
        point_count=len(input_deg),
    )


def with_target_points(
    inputs: LoadedInputs,
    input_angles_deg: Sequence[float],
    output_angles_deg: Sequence[float],
    weights: Sequence[float],
) -> LoadedInputs:
    """Return a copy of ``inputs`` with the target
    curve replaced by user-edited support points.

    Angles are given in DEGREES (user units) and
    converted to radians; weights are non-negative
    and clamped to ``MAX_WEIGHT``.

    Raises :class:`WorkflowError` with a user-facing
    message when the points are invalid (fewer than
    two points, unsorted input angles, negative
    weights).
    """

    try:
        target_curve = (
            TargetCurve.from_points_strict(
                tuple(
                    math.radians(angle)
                    for angle in input_angles_deg
                ),
                tuple(
                    math.radians(angle)
                    for angle in output_angles_deg
                ),
                tuple(
                    float(weight)
                    for weight in weights
                ),
            )
        )
    except ValueError as error:
        raise WorkflowError(
            f"Ungültige Zielkurve: {error}",
        ) from error
    return replace(
        inputs,
        target_curve=target_curve,
        input_angles=target_curve.input_angles,
        output_angles=target_curve.output_angles,
        point_count=len(
            target_curve.input_angles,
        ),
    )


def with_mechanism_levers(
    inputs: LoadedInputs,
    lever_rows: Sequence[Mapping[str, object]],
) -> LoadedInputs:
    """Return a copy of ``inputs`` with the mechanism
    definition replaced by user-edited lever rows.

    Each row is a mapping with the keys ``id``,
    ``length_min``, ``length_max``, ``length_start``,
    ``angle_min``, ``angle_max``, ``angle_start``,
    ``pivot_x_start``/``pivot_x``,
    ``pivot_y_start``/``pivot_y``,
    ``pivot_z``, ``axis_x``,
    ``axis_y``, ``axis_z`` and optionally ``ref_x``,
    ``ref_y``, ``ref_z``, ``driver``, ``coupled`` and
    ``pivot_on``.

    Lengths are in millimetres, angles in DEGREES
    (user units) and are converted to radians.
    ``driver``/``coupled`` accept integers or ``None``;
    ``ref_x``/``ref_y``/``ref_z`` accept floats or
    ``None`` (automatic reference direction).  Empty
    cells from an editable UI table arrive as ``None``
    and are treated as absent.

    Raises :class:`WorkflowError` with a user-facing
    message when a row is invalid (missing values,
    malformed integers, unknown pivot_on reference,
    duplicate lever ids).
    """

    def _number(
        row: Mapping[str, object],
        key: str,
        *,
        fallback: str | None = None,
    ) -> float:
        value = row.get(key)
        if value is None and fallback is not None:
            value = row.get(fallback)
        if value is None or (
            isinstance(value, str)
            and not value.strip()
        ):
            raise WorkflowError(
                f"Hebel {row.get('id', '?')}: "
                f"Feld {key!r} fehlt."
            )
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise WorkflowError(
                f"Hebel {row.get('id', '?')}: "
                f"Feld {key!r} ist keine Zahl "
                f"({value!r})."
            ) from error

    def _optional_int(
        row: Mapping[str, object],
        key: str,
    ) -> int | None:
        value = row.get(key)
        if value is None or (
            isinstance(value, str)
            and not value.strip()
        ):
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise WorkflowError(
                f"Hebel {row.get('id', '?')}: "
                f"Feld {key!r} ist keine ganze "
                f"Zahl ({value!r})."
            ) from error

    def _pivot_on(
        row: Mapping[str, object],
    ) -> str | None:
        value = row.get("pivot_on")
        if value is None or (
            isinstance(value, str)
            and not value.strip()
        ):
            return None
        try:
            parse_pivot_reference(str(value))
        except ValueError as error:
            raise WorkflowError(
                f"Hebel {row.get('id', '?')}: "
                f"{error}"
            ) from error
        return str(value).strip()

    levers = []
    for row in lever_rows:
        reference = None
        if not all(
            row.get(key) is None
            or (
                isinstance(row.get(key), str)
                and not str(row.get(key)).strip()
            )
            for key in ("ref_x", "ref_y", "ref_z")
        ):
            reference = Vector3D(
                x=_number(row, "ref_x"),
                y=_number(row, "ref_y"),
                z=_number(row, "ref_z"),
            )
        lever_id = _optional_int(row, "id")
        if lever_id is None:
            raise WorkflowError(
                "Jeder Hebel benötigt eine "
                "numerische id."
            )
        coupled = _optional_int(row, "coupled")
        driver = _optional_int(row, "driver")
        if coupled is not None:
            driver = None
        try:
            levers.append(
                LeverDefinition(
                    id=lever_id,
                    pivot=Point3D(
                        x=_number(
                            row,
                            "pivot_x_start",
                            fallback="pivot_x",
                        ),
                        y=_number(
                            row,
                            "pivot_y_start",
                            fallback="pivot_y",
                        ),
                        z=_number(row, "pivot_z"),
                    ),
                    length_min=_number(
                        row,
                        "length_min",
                    ),
                    length_max=_number(
                        row,
                        "length_max",
                    ),
                    length_start=_number(
                        row,
                        "length_start",
                    ),
                    angle_min=math.radians(
                        _number(row, "angle_min")
                    ),
                    angle_max=math.radians(
                        _number(row, "angle_max")
                    ),
                    angle_start=math.radians(
                        _number(row, "angle_start")
                    ),
                    axis=Vector3D(
                        x=_number(row, "axis_x"),
                        y=_number(row, "axis_y"),
                        z=_number(row, "axis_z"),
                    ),
                    reference_direction=reference,
                    driver=driver,
                    coupled=coupled,
                    pivot_on=_pivot_on(row),
                )
            )
        except ValueError as error:
            raise WorkflowError(
                f"Ungültige Mechanikdefinition: "
                f"{error}"
            ) from error

    try:
        definition = MechanismDefinition(
            tuple(levers)
        )
    except ValueError as error:
        raise WorkflowError(
            f"Ungültige Mechanikdefinition: {error}"
        ) from error
    if not definition.levers:
        raise WorkflowError(
            "Die Mechanikdefinition benötigt "
            "mindestens einen Hebel."
        )

    return replace(
        inputs,
        definition=definition,
    )


def optimized_values(
    definition: MechanismDefinition,
) -> dict[str, float]:
    """Map optimization parameter names to the values
    of a mechanism definition (INTERNAL units:
    radians for angles, millimetres for lengths).

    The keys match the names produced by
    ``CsvParameterFactory`` (``lever.<id>.length``,
    ``lever.<id>.angle``, ``lever.<id>.pivot.x`` and
    ``lever.<id>.pivot.y``), so an optimized
    definition can be compared with the start
    template parameter by parameter.
    """

    values: dict[str, float] = {}
    for lever in definition.levers:
        values[
            f"lever.{lever.id}.length"
        ] = lever.length_start
        values[
            f"lever.{lever.id}.angle"
        ] = lever.angle_start
        values[
            f"lever.{lever.id}.pivot.x"
        ] = lever.pivot.x
        values[
            f"lever.{lever.id}.pivot.y"
        ] = lever.pivot.y
    return values


def parameter_template(
    inputs: LoadedInputs,
) -> ParameterSet:
    """Create the editable optimization parameter
    template from the mechanism definition."""

    return CsvParameterFactory.create(
        inputs.definition,
    )


def is_angle_parameter(
    parameter: Parameter,
) -> bool:
    """True when the parameter is an angle (shown in
    degrees in the UI)."""

    return parameter.name.endswith(".angle")


def apply_parameter_overrides(
    template: ParameterSet,
    overrides: dict[
        str,
        tuple[float, float, float],
    ],
) -> ParameterSet:
    """Apply user edits to the parameter template.

    ``overrides`` maps a parameter name to
    ``(minimum, maximum, value)`` in USER units
    (degrees for angle parameters, millimetres for
    length parameters).
    """

    parameters = []
    for parameter in template.parameters:
        if parameter.name in overrides:
            minimum, maximum, value = overrides[
                parameter.name
            ]
            if is_angle_parameter(parameter):
                minimum = math.radians(minimum)
                maximum = math.radians(maximum)
                value = math.radians(value)
            parameter = replace(
                parameter,
                minimum=minimum,
                maximum=maximum,
                value=value,
            )
        parameters.append(parameter)
    return ParameterSet(
        parameters=tuple(parameters),
    )


def optimized_display_values(
    rows: Sequence[Mapping[str, object]],
    optimized_map: Mapping[str, float],
    snapshot: Mapping[int, Mapping[str, float]],
) -> dict[
    int,
    tuple[float | None, ...],
]:
    """Optimized values displayed per lever id.

    A value is only displayed while the
    corresponding start value still matches the
    snapshot taken when the optimization run
    started; otherwise the cell is cleared
    (``None``).  Lengths are in millimetres,
    angles in DEGREES (user units).

    The returned tuple is ordered
    ``(length, angle, pivot_x, pivot_y)``; rows
    without pivot columns keep ``None`` for the
    pivot entries.
    """

    result: dict[
        int,
        tuple[float | None, ...],
    ] = {}
    for row in rows:
        if row.get("id") is None:
            continue
        lever_id = int(row["id"])
        length_key = (
            f"lever.{lever_id}.length"
        )
        angle_key = (
            f"lever.{lever_id}.angle"
        )
        pivot_x_key = (
            f"lever.{lever_id}.pivot.x"
        )
        pivot_y_key = (
            f"lever.{lever_id}.pivot.y"
        )
        lever_snapshot = snapshot.get(
            lever_id,
        )
        length_value = (
            optimized_map[length_key]
            if lever_snapshot is not None
            and lever_snapshot["length"]
            == row["length_start"]
            and length_key in optimized_map
            else None
        )
        angle_value = (
            math.degrees(
                optimized_map[angle_key],
            )
            if lever_snapshot is not None
            and lever_snapshot["angle"]
            == row["angle_start"]
            and angle_key in optimized_map
            else None
        )
        pivot_x_value = (
            optimized_map[pivot_x_key]
            if lever_snapshot is not None
            and lever_snapshot.get(
                "pivot_x"
            )
            == row.get("pivot_x_start")
            and pivot_x_key in optimized_map
            else None
        )
        pivot_y_value = (
            optimized_map[pivot_y_key]
            if lever_snapshot is not None
            and lever_snapshot.get(
                "pivot_y"
            )
            == row.get("pivot_y_start")
            and pivot_y_key in optimized_map
            else None
        )
        result[lever_id] = (
            length_value,
            angle_value,
            pivot_x_value,
            pivot_y_value,
        )
    return result


def adopt_single_optimized_values(
    rows: list[dict],
) -> bool:
    """Adopt individually selected optimized
    values as new start values, in place.

    Each row may carry ``length_adopt`` /
    ``angle_adopt`` flags (and, when the pivot
    columns are shown, ``pivot_x_adopt`` /
    ``pivot_y_adopt``) set by the user.  For
    every set flag whose corresponding
    ``*_optimized`` cell is not empty, the
    optimized value is written into the
    corresponding ``*_start`` field and the flag
    is cleared.  Returns ``True`` when at least
    one value was adopted (the caller then
    reruns so the table reflects the adoption).
    """

    def _adopt(
        row: dict,
        adopt_key: str,
        optimized_key: str,
        start_key: str,
    ) -> bool:
        if (
            row.get(adopt_key)
            and row.get(optimized_key)
            is not None
        ):
            row[start_key] = row[
                optimized_key
            ]
            row[adopt_key] = False
            return True
        return False

    adopted = False
    for row in rows:
        if row.get("id") is None:
            continue
        if _adopt(
            row,
            "length_adopt",
            "length_optimized",
            "length_start",
        ):
            adopted = True
        if _adopt(
            row,
            "angle_adopt",
            "angle_optimized",
            "angle_start",
        ):
            adopted = True
        if _adopt(
            row,
            "pivot_x_adopt",
            "pivot_x_optimized",
            "pivot_x_start",
        ):
            adopted = True
        if _adopt(
            row,
            "pivot_y_adopt",
            "pivot_y_optimized",
            "pivot_y_start",
        ):
            adopted = True
    return adopted


def _simulate(
    builder: CsvMechanismBuilder,
    candidate: ParameterSet,
    inputs: LoadedInputs,
):
    simulator = _anchored_simulator(
        inputs,
        candidate,
    )
    mechanism = builder.build(candidate)
    results = simulator.simulate(mechanism)
    if not all(
        result.success for result in results
    ):
        raise WorkflowError(
            "Die Mechanik blockiert im "
            "Bewegungsbereich; es gibt keine "
            "vollständige Simulation.",
        )
    return mechanism, simulator, results


def build_report_html(
    mechanism,
    simulator: MechanismSimulator,
    results,
    inputs: LoadedInputs,
    *,
    title: str = (
        "TraSim — Ergebnis"
    ),
    actual_label: str = (
        "Ist-Ausgangskurve"
    ),
) -> str:
    """Build the usual HTML result document for a
    simulated mechanism.

    Start angle semantics: the Soll (target) output
    angles are anchored at the output lever's start
    angle (its reference angle in ``mechanism``,
    which reflects the applied angle parameter) at
    the middle support position, and the Ist
    (actual) curve is drawn over the anchored input
    angles.
    """

    plotter = CurvePlotter(title=title)
    final_result = results[-1]
    drive_result = results[0]
    output_angle = (
        mechanism.stages[-1].output_angle
        if mechanism.stages
        else middle_output_angle(inputs)
    )
    input_offset = (
        drive_result.input_angles[0]
        - inputs.input_angles[0]
    )
    output_offset = (
        output_angle
        - middle_output_angle(inputs)
    )
    plotter.set_target_curve(
        inputs.target_curve,
        input_angles=inputs.input_angles,
        output_angles=inputs.output_angles,
    )
    plotter.add_actual_curve(
        tuple(
            angle - input_offset
            for angle in drive_result.input_angles
        ),
        tuple(
            angle - output_offset
            for angle in final_result.output_angles
        ),
        label=actual_label,
    )
    for state in mechanism_states_from_results(
        mechanism,
        results,
    ):
        plotter.add_mechanism_state(state)
    return plotter.build_html()


def preview(
    inputs: LoadedInputs,
    template: ParameterSet | None = None,
) -> str:
    """Simulate the start mechanism and return the
    HTML report of the unoptimized state.

    ``template`` overrides the parameter template
    derived from the definition (e.g. to honour
    user-edited start values and bounds)."""

    builder = CsvMechanismBuilder(
        inputs.definition,
    )
    if template is None:
        template = parameter_template(inputs)
    mechanism, simulator, results = _simulate(
        builder,
        template,
        inputs,
    )
    return build_report_html(
        mechanism,
        simulator,
        results,
        inputs,
        title=(
            "TraSim — Vorschau "
            "(Startmechanik)"
        ),
    )


def mechanism_csv_text(
    definition: MechanismDefinition,
    convention: CsvConvention,
) -> str:
    """Serialize a mechanism definition to CSV
    text."""

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "mechanism.csv"
        CsvWriter.write_mechanism(
            definition,
            path,
            convention=convention,
        )
        return path.read_text(
            encoding="utf-8",
        )


def _build_optimizer(
    inputs: LoadedInputs,
    settings: OptimizationSettings,
) -> Callable[[ParameterSet], float]:
    """
    Build a fresh evaluator (builder/simulator/fitness)
    in the calling process.

    In parallel mode this runs once per worker
    process as the pool initializer's
    evaluator_factory; in sequential mode it is
    used directly.  It must live at module level
    so Windows spawn can pickle it.

    Start angle semantics: the drive lever's angle
    parameter is the position it occupies at the
    middle support position of the target curve.
    The support-point input angles are therefore
    shifted by ``(drive angle - middle input
    angle)`` for every candidate, and the target
    curve's support input angles are shifted by
    the same constant so the fitness still
    compares output angles support point by
    support point.
    """

    builder = CsvMechanismBuilder(
        inputs.definition,
    )
    drive_id = _drive_lever_id(
        inputs.definition,
    )
    output_id = _output_lever_id(
        inputs.definition,
    )
    drive_fallback = (
        inputs.definition.get_lever(
            drive_id,
        ).angle_start
    )
    output_fallback = (
        inputs.definition.get_lever(
            output_id,
        ).angle_start
    )
    weights = inputs.target_curve.weights
    gradient_weight = settings.gradient_weight

    def evaluate(
        candidate: ParameterSet,
    ) -> float:

        values = candidate.values()
        drive_angle = values.get(
            f"lever.{drive_id}.angle",
            drive_fallback,
        )
        output_angle = values.get(
            f"lever.{output_id}.angle",
            output_fallback,
        )
        angles = anchored_input_angles(
            inputs,
            drive_angle,
        )
        outputs = anchored_output_angles(
            inputs,
            output_angle,
        )
        target = (
            TargetCurve.from_points_strict(
                angles,
                outputs,
                weights,
            )
        )
        fitness = CurveFitness(
            target_curve=target,
            motion_start=angles[0],
            motion_range=(
                angles[-1] - angles[0]
            ),
            gradient_weight=(
                gradient_weight
            ),
        )
        simulator = MechanismSimulator(
            motion=PointMotion(
                angles=angles,
            ),
            stage_simulator=StageSimulator(),
            stage_limit=None,
        )
        return MechanismOptimizer(
            builder=builder,
            simulator=simulator,
            fitness=fitness,
        ).evaluate(candidate)

    return evaluate


def run_optimization(
    inputs: LoadedInputs,
    template: ParameterSet,
    settings: OptimizationSettings | None = None,
    progress: (
        Callable[[int, float], None] | None
    ) = None,
) -> OptimizationResult:
    """Run the evolutionary optimization.

    ``progress(generation, best_score)`` is called
    once per generation so UIs can show live
    progress.

    Raises :class:`WorkflowError` with a user-facing
    message when the optimization cannot produce a
    result.
    """

    if settings is None:
        settings = OptimizationSettings()

    worker_count = settings.workers
    if worker_count <= 0:
        worker_count = multiprocessing.cpu_count()
    worker_count = max(1, worker_count)

    evaluator = _build_optimizer(
        inputs,
        settings,
    )

    rng = random.Random(settings.seed)
    population = PopulationFactory(
        random_generator=rng,
    ).create(
        template,
        size=settings.population_size,
    )
    engine = EvolutionEngine(
        population=population,
        evaluator=evaluator if worker_count == 1 else None,
        evaluator_factory=(
            partial(
                _build_optimizer,
                inputs,
                settings,
            )
            if worker_count > 1
            else None
        ),
        workers=worker_count,
        selection_count=settings.selection_count,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                strength=(
                    settings.mutation_strength
                ),
                random_generator=rng,
            ),
            random_generator=rng,
        ),
        target_fitness=settings.target_fitness,
        max_generations=(
            settings.max_generations
        ),
        stagnation_limit=(
            settings.stagnation_limit
        ),
        stagnation_tolerance=(
            settings.stagnation_tolerance
        ),
    )

    engine.evaluate_population()
    valid = sum(
        score < float("inf")
        for score in engine.scores.values()
    )
    if valid == 0:
        raise WorkflowError(
            "Keine gültigen Startkandidaten — "
            "die Mechanik ist im Suchraum "
            "unzulässig. Bitte Parameterbereiche "
            "erweitern.",
        )

    history: list[float] = [engine.best_score]
    generations = 0
    for generation in engine.run(
        children_count=(
            settings.children_count
        ),
    ):
        generations += 1
        history.append(engine.best_score)
        if progress is not None:
            progress(
                generations,
                engine.best_score,
            )

    best = engine.best_candidate
    if best is None:
        raise WorkflowError(
            "Die Optimierung hat keinen "
            "Kandidaten erzeugt.",
        )

    best_values = best.values()
    optimized_definition = replace(
        inputs.definition,
        levers=tuple(
            replace(
                lever,
                length_start=best_values.get(
                    f"lever.{lever.id}.length",
                    lever.length_start,
                ),
                angle_start=best_values.get(
                    f"lever.{lever.id}.angle",
                    lever.angle_start,
                ),
            )
            for lever in inputs.definition.levers
        ),
    )

    final_builder = CsvMechanismBuilder(
        inputs.definition,
    )
    final_simulator = _anchored_simulator(
        inputs,
        best,
    )
    mechanism = final_builder.build(best)
    results = final_simulator.simulate(mechanism)
    if not all(
        result.success for result in results
    ):
        raise WorkflowError(
            "Die optimierte Mechanik blockiert "
            "im Bewegungsbereich.",
        )
    html = build_report_html(
        mechanism,
        final_simulator,
        results,
        inputs,
        title=(
            "TraSim — Optimiertes Getriebe"
        ),
    )

    return OptimizationResult(
        best_score=engine.best_score,
        stop_reason=(
            engine.stop_reason or ""
        ),
        generations=generations,
        optimized_definition=(
            optimized_definition
        ),
        mechanism_csv=mechanism_csv_text(
            optimized_definition,
            inputs.output_convention,
        ),
        html=html,
        fitness_history=tuple(history),
    )
