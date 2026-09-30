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
"""

from __future__ import annotations

import csv
import math
import random
import tempfile
from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
    replace,
)
from pathlib import Path

from analysis.curve_fitness import (
    CurveFitness,
)
from analysis.curve_plotter import (
    CurvePlotter,
    mechanism_states_from_results,
)
from analysis.target_curve import (
    TargetCurve,
)
from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)
from mechanism_io.csv_convention import (
    GERMAN,
    CsvConvention,
    detect_convention,
)
from mechanism_io.csv_reader import (
    CsvReader,
)
from mechanism_io.csv_writer import (
    CsvWriter,
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
    seed: int = 42


@dataclass(frozen=True, slots=True)
class LoadedInputs:
    """Parsed and validated user inputs."""

    mechanism_convention: CsvConvention
    target_convention: CsvConvention
    output_convention: CsvConvention
    definition: MechanismDefinition
    target_curve: TargetCurve
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
) -> tuple[list[float], list[float]]:
    input_angles: list[float] = []
    output_angles: list[float] = []
    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file,
            delimiter=convention.delimiter,
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
    if len(input_angles) < 2:
        raise WorkflowError(
            "Die Zielkurve benötigt mindestens "
            "zwei Stützpunkte.",
        )
    return input_angles, output_angles


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
            input_deg, output_deg = _read_target_points(
                target_path,
                target_convention,
            )
            if not definition.levers:
                raise WorkflowError(
                    "Die Mechanik-Datei enthält "
                    "keine Hebel. Bitte das "
                    "CSV-Format prüfen "
                    "(Spalten wie id, length_min, "
                    "…, driver, coupled).",
                )
            target_curve = TargetCurve.from_points(
                tuple(input_deg),
                tuple(output_deg),
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


def _simulate(
    builder: CsvMechanismBuilder,
    candidate: ParameterSet,
    inputs: LoadedInputs,
):
    simulator = MechanismSimulator(
        motion=PointMotion(
            angles=inputs.input_angles,
        ),
        stage_simulator=StageSimulator(),
        stage_limit=None,
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
    simulated mechanism."""

    plotter = CurvePlotter(title=title)
    plotter.set_target_curve(
        inputs.target_curve,
        input_angles=inputs.input_angles,
        output_angles=inputs.output_angles,
    )
    final_result = results[-1]
    drive_result = results[0]
    plotter.add_actual_curve(
        drive_result.input_angles,
        final_result.output_angles,
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
) -> str:
    """Simulate the start mechanism and return the
    HTML report of the unoptimized state."""

    builder = CsvMechanismBuilder(
        inputs.definition,
    )
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
    builder = CsvMechanismBuilder(
        inputs.definition,
    )
    simulator = MechanismSimulator(
        motion=PointMotion(
            angles=inputs.input_angles,
        ),
        stage_simulator=StageSimulator(),
        stage_limit=None,
    )
    fitness = CurveFitness(
        target_curve=inputs.target_curve,
        motion_start=inputs.input_angles[0],
        motion_range=(
            inputs.input_angles[-1]
            - inputs.input_angles[0]
        ),
    )
    optimizer = MechanismOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
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
        evaluator=optimizer.evaluate,
        selection_count=settings.selection_count,
        reproduction=Reproduction(
            mutation=ParameterMutation(
                strength=(
                    settings.mutation_strength
                ),
                random_generator=rng,
            ),
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

    mechanism = builder.build(best)
    results = simulator.simulate(mechanism)
    if not all(
        result.success for result in results
    ):
        raise WorkflowError(
            "Die optimierte Mechanik blockiert "
            "im Bewegungsbereich.",
        )
    html = build_report_html(
        mechanism,
        simulator,
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
