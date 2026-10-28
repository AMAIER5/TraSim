"""gui/mixed_brake_workflow.py

End-to-end workflow for the MIXED BRAKE (mixer) mode:
mechanism CSV with a ``pivot_on`` mixer lever, two base target
curves (unbraked / fully braked), an evenly spaced brake grid
(0/25/50/75/100 % brake travel) with blended derivative target
curves, the soft minimum transmission angle criterion and the
evolutionary optimization over ALL brake positions.

The module is independent of any particular UI framework;
``gui/app.py`` (Streamlit) is a thin layer on top of it and the
unit tests exercise the workflow directly.

User I/O angles are in DEGREES, internal calculations use
RADIANS.
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from dataclasses import (
    dataclass,
    field,
)

from analysis.brake_curve_fitness import (
    BrakeCurveFitness,
)
from analysis.brake_grid import (
    BrakeGridSettings,
    blend_target_curves,
    make_brake_grid,
)
from analysis.curve_plotter import (
    CurvePlotter,
    MechanismPlotState,
    mechanism_states_from_results,
)
from analysis.endpoint_constraint import (
    EndpointConstraint,
)
from analysis.target_curve import TargetCurve
from core.point3d import Point3D
from analysis.transmission_angle import (
    TransmissionAngleConstraint,
    min_transmission_angle,
)
from gui.workflow import (
    WorkflowError,
    _text_to_temp_file,
)
from mechanics.mixed_brake_builder import (
    MechanismBuildError,
    MixedBrakeBuilder,
)
from mechanism_io.csv_reader import CsvReader
from mechanism_io.csv_writer import CsvWriter
from model.mechanism_definition import (
    MechanismDefinition,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
)
from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.mixed_brake_optimizer import (
    MixedBrakeOptimizer,
)
from optimization.parameter_mutation import (
    ParameterMutation,
)
from optimization.parameter_set import ParameterSet
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import Reproduction
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange

__all__ = [
    "MixedBrakeInputs",
    "MixedBrakeResult",
    "MixedBrakeSettings",
    "MixedBrakeState",
    "build_mixed_brake_grid",
    "build_mixed_brake_state",
    "load_mixed_brake_inputs",
    "mixed_brake_parameter_template",
    "mixed_brake_preview",
    "run_mixed_brake_optimization",
]


@dataclass(frozen=True, slots=True)
class MixedBrakeSettings:
    """Settings of the mixed brake optimization."""

    population_size: int = 40
    children_count: int = 40
    selection_count: int = 12
    mutation_strength: float = 0.03
    max_generations: int = 40
    stagnation_limit: int = 20
    stagnation_tolerance: float = 1e-8
    target_fitness: float = 0.005
    brake_positions: int = 5
    transmission_angle_deg: float = 20.0
    transmission_weight: float = 1.0
    seed: int = 42


@dataclass(frozen=True, slots=True)
class MixedBrakeInputs:
    """Parsed and validated mixed brake inputs."""

    definition: MechanismDefinition
    unbraked: TargetCurve
    braked: TargetCurve
    motion_start_deg: float
    motion_travel_deg: float
    motion_step_deg: float
    brake_min_deg: float
    brake_max_deg: float


@dataclass(frozen=True, slots=True)
class MixedBrakeResult:
    """Outcome of a completed mixed brake optimization."""

    best_score: float
    stop_reason: str
    generations: int
    brake_positions_deg: tuple[float, ...]
    fitness_history: tuple[float, ...]
    best_values: dict[str, float]
    html: str
    min_transmission_angle_deg: float = field(
        default=0.0,
    )
    optimized_definition: (
        MechanismDefinition | None
    ) = None
    mechanism_csv: str = ""


@dataclass(frozen=True, slots=True)
class MixedBrakeState:
    """Visualization state of one brake position."""

    brake_fraction: float
    brake_angle_deg: float
    input_angles_deg: tuple[float, ...]
    target_deg: tuple[float, ...]
    actual_deg: tuple[float, ...]
    states: tuple[
        MechanismPlotState,
        MechanismPlotState,
        MechanismPlotState,
    ]


@dataclass(frozen=True, slots=True)
class _MixedBrakeGrid:
    """Grid of one blended target curve and one
    simulated mechanism state per brake position."""

    fractions: tuple[float, ...]
    brake_positions_deg: tuple[float, ...]
    input_angles_deg: tuple[float, ...]
    targets_deg: (
        tuple[tuple[float, ...], ...]
    )
    states: (
        tuple[MixedBrakeState | None, ...]
    )
    min_transmission_angle_deg: float


def load_mixed_brake_inputs(
    mechanism_csv: str,
    unbraked_csv: str,
    braked_csv: str,
    *,
    motion_start_deg: float = -10.0,
    motion_travel_deg: float = 20.0,
    motion_step_deg: float = 2.0,
) -> MixedBrakeInputs:
    """Load and validate the mixed brake inputs from CSV
    text.

    Raises WorkflowError with a user-facing message
    on invalid input.
    """
    try:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            mechanism_path = _text_to_temp_file(
                directory,
                "mechanism.csv",
                mechanism_csv,
            )
            unbraked_path = _text_to_temp_file(
                directory,
                "target_curve_unbraked.csv",
                unbraked_csv,
            )
            braked_path = _text_to_temp_file(
                directory,
                "target_curve_braked.csv",
                braked_csv,
            )
            return _load_mixed_brake_inputs(
                mechanism_path,
                unbraked_path,
                braked_path,
                motion_start_deg=motion_start_deg,
                motion_travel_deg=motion_travel_deg,
                motion_step_deg=motion_step_deg,
            )
    except WorkflowError:
        raise
    except Exception as error:
        raise WorkflowError(
            f"Eingabedatei konnte nicht gelesen "
            f"werden: {error}",
        ) from error


def mixed_brake_inputs_from_parts(
    definition: MechanismDefinition,
    unbraked: TargetCurve,
    braked: TargetCurve,
    *,
    motion_start_deg: float = -10.0,
    motion_travel_deg: float = 20.0,
    motion_step_deg: float = 2.0,
) -> MixedBrakeInputs:
    """Build mixed brake inputs from an (edited)
    mechanism definition and the two base target
    curves.

    Raises WorkflowError when the definition does not
    contain exactly one mixer lever with pivot_on.
    """
    mixers = [
        lever
        for lever in definition.levers
        if lever.pivot_on is not None
    ]
    if len(mixers) != 1:
        raise WorkflowError(
            "Der Mischer-Modus ben\u00f6tigt genau einen "
            "Hebel mit pivot_on (Mischerhebel); "
            f"gefunden: {len(mixers)}."
        )
    mixer = mixers[0]
    brake = definition.get_lever(
        mixer.pivot_reference.lever_id
    )
    return MixedBrakeInputs(
        definition=definition,
        unbraked=unbraked,
        braked=braked,
        motion_start_deg=motion_start_deg,
        motion_travel_deg=motion_travel_deg,
        motion_step_deg=motion_step_deg,
        brake_min_deg=math.degrees(
            brake.angle_start
        ),
        brake_max_deg=math.degrees(
            brake.angle_max
        ),
    )


def _load_mixed_brake_inputs(
    mechanism_path,
    unbraked_path,
    braked_path,
    *,
    motion_start_deg: float,
    motion_travel_deg: float,
    motion_step_deg: float,
) -> MixedBrakeInputs:
    try:
        definition = CsvReader.read_mechanism(
            mechanism_path,
        )
        mixers = [
            lever
            for lever in definition.levers
            if lever.pivot_on is not None
        ]
        if len(mixers) != 1:
            raise WorkflowError(
                "Der Mischer-Modus benötigt genau einen "
                "Hebel mit pivot_on (Mischerhebel); "
                f"gefunden: {len(mixers)}."
            )
        mixer = mixers[0]
        brake = definition.get_lever(
            mixer.pivot_reference.lever_id
        )
        unbraked = TargetCurve.from_csv(
            unbraked_path,
        )
        braked = TargetCurve.from_csv(
            braked_path,
        )
    except WorkflowError:
        raise
    except Exception as error:
        raise WorkflowError(
            f"Eingabedatei konnte nicht gelesen "
            f"werden: {error}",
        ) from error
    return MixedBrakeInputs(
        definition=definition,
        unbraked=unbraked,
        braked=braked,
        motion_start_deg=motion_start_deg,
        motion_travel_deg=motion_travel_deg,
        motion_step_deg=motion_step_deg,
        brake_min_deg=math.degrees(
            brake.angle_start
        ),
        brake_max_deg=math.degrees(
            brake.angle_max
        ),
    )


def mixed_brake_parameter_template(
    inputs: MixedBrakeInputs,
) -> ParameterSet:
    """Editable optimization parameter template of the
    mixer mechanism."""
    return CsvParameterFactory.create(
        inputs.definition,
    )


def _motion_range(
    inputs: MixedBrakeInputs,
) -> MotionRange:
    return MotionRange(
        start_angle=math.radians(
            inputs.motion_start_deg
        ),
        max_angle=math.radians(
            inputs.motion_travel_deg
        ),
        step=math.radians(
            inputs.motion_step_deg
        ),
    )


def _parameter_definition(
    inputs: MixedBrakeInputs,
    parameters: ParameterSet,
) -> MechanismDefinition:
    """Apply optimization parameters (length, angle,
    pivot) onto the input definition."""
    from dataclasses import replace

    values = parameters.values()
    levers = []
    for lever in inputs.definition.levers:
        levers.append(
            replace(
                lever,
                length_start=values.get(
                    f"lever.{lever.id}.length",
                    lever.length_start,
                ),
                angle_start=values.get(
                    f"lever.{lever.id}.angle",
                    lever.angle_start,
                ),
                pivot=(
                    Point3D(
                        x=values.get(
                            f"lever.{lever.id}.pivot.x",
                            lever.pivot.x,
                        ),
                        y=values.get(
                            f"lever.{lever.id}.pivot.y",
                            lever.pivot.y,
                        ),
                        z=lever.pivot.z,
                    )
                    if lever.pivot_on is None
                    else lever.pivot
                ),
            )
        )
    return MechanismDefinition(tuple(levers))


def build_mixed_brake_grid(
    inputs: MixedBrakeInputs,
    definition: MechanismDefinition,
    *,
    position_count: int = 5,
    motion: MotionRange | None = None,
) -> _MixedBrakeGrid:
    """Build the mechanism for every brake position of
    the evenly spaced brake grid and simulate it over
    the motion range.

    Positions whose mechanism cannot be built (for
    example because a candidate is not mountable there)
    yield a ``None`` entry in ``states`` instead of
    raising, so reports can still show the remaining
    positions.
    """
    grid = BrakeGridSettings(
        brake_min_deg=inputs.brake_min_deg,
        brake_max_deg=inputs.brake_max_deg,
        position_count=position_count,
    )
    if motion is None:
        motion = _motion_range(inputs)
    input_angles = tuple(
        angle for angle in motion
    )
    targets = blend_target_curves(
        inputs.unbraked,
        inputs.braked,
        grid.fractions,
        input_angles=input_angles,
    )
    builder = MixedBrakeBuilder(definition)
    simulator = MechanismSimulator(motion=motion)
    targets_deg = tuple(
        tuple(
            math.degrees(angle)
            for angle in target.sample(
                input_angles,
            ).output_angles
        )
        for target in targets
    )
    states: list[MixedBrakeState | None] = []
    minimum_deg = float("inf")
    input_deg = tuple(
        round(math.degrees(angle), 1)
        for angle in input_angles
    )
    for (
        fraction,
        beta_deg,
        target_deg,
    ) in zip(
        grid.fractions,
        grid.brake_positions_deg,
        targets_deg,
    ):
        try:
            mechanism = builder.build(
                math.radians(beta_deg)
            )
            results = simulator.simulate(
                mechanism
            )
        except MechanismBuildError:
            states.append(None)
            continue
        if not results or not all(
            result.success
            for result in results
        ):
            states.append(None)
            continue
        minimum_deg = min(
            minimum_deg,
            math.degrees(
                min_transmission_angle(
                    mechanism
                )
            ),
        )
        states.append(
            MixedBrakeState(
                brake_fraction=fraction,
                brake_angle_deg=beta_deg,
                input_angles_deg=input_deg,
                target_deg=target_deg,
                actual_deg=tuple(
                    round(math.degrees(angle), 1)
                    for angle in results[
                        -1
                    ].output_angles
                ),
                states=mechanism_states_from_results(
                    mechanism,
                    results,
                ),
            )
        )
    return _MixedBrakeGrid(
        fractions=grid.fractions,
        brake_positions_deg=(
            grid.brake_positions_deg
        ),
        input_angles_deg=input_deg,
        targets_deg=targets_deg,
        states=tuple(states),
        min_transmission_angle_deg=(
            minimum_deg
        ),
    )


def build_mixed_brake_state(
    inputs: MixedBrakeInputs,
    parameters: ParameterSet,
    *,
    brake_fraction: float,
    position_count: int = 5,
    motion: MotionRange | None = None,
) -> MixedBrakeState:
    """Simulate one brake position of a (candidate)
    parameter set.

    ``brake_fraction`` is the brake travel in [0, 1]
    (0 % = unbraked, 100 % = fully braked); it is
    snapped onto the evenly spaced brake grid.

    Raises WorkflowError with a user-facing message
    when the mechanism cannot be built or simulated
    for the requested position.
    """
    if not 0.0 <= brake_fraction <= 1.0:
        raise WorkflowError(
            "Die Bremsstellung muss zwischen 0 % "
            "und 100 % liegen."
        )
    grid = build_mixed_brake_grid(
        inputs,
        _parameter_definition(
            inputs,
            parameters,
        ),
        position_count=position_count,
        motion=motion,
    )
    count = len(grid.fractions)
    index = int(
        min(
            max(
                round(
                    brake_fraction
                    * (count - 1)
                ),
                0,
            ),
            count - 1,
        )
    )
    state = grid.states[index]
    if state is None:
        raise WorkflowError(
            "Die Mechanik kann f\u00fcr die gew\u00e4hlte "
            "Bremsstellung nicht aufgebaut werden "
            "(Geometrie unzul\u00e4ssig oder blockiert)."
        )
    return state


def _brake_position_report(
    grid: _MixedBrakeGrid,
    *,
    title: str,
) -> str:
    """HTML report showing all blended target curves
    and actual curves plus the mechanism snapshots of
    the middle brake position."""
    plotter = CurvePlotter(title=title)
    middle = len(grid.fractions) // 2
    middle_state = (
        grid.states[middle]
        if grid.states
        else None
    )
    for state in grid.states:
        if state is None:
            continue
        fraction = state.brake_fraction
        input_deg = state.input_angles_deg
        plotter.add_target_curve(
            input_deg,
            state.target_deg,
            label=(
                f"Soll Bremse "
                f"{100 * fraction:.0f} %"
            ),
        )
        plotter.add_actual_curve(
            input_deg,
            state.actual_deg,
            label=(
                f"Bremse "
                f"{100 * fraction:.0f} %"
            ),
        )
    if middle_state is not None:
        for state in middle_state.states:
            plotter.add_mechanism_state(
                state
            )
    return plotter.build_html()


def mixed_brake_preview(
    inputs: MixedBrakeInputs,
    template: ParameterSet | None = None,
    *,
    position_count: int = 5,
) -> str:
    """Simulate the start mechanism over all brake
    positions and return the HTML preview report."""
    if template is None:
        template = (
            mixed_brake_parameter_template(inputs)
        )
    grid = build_mixed_brake_grid(
        inputs,
        _parameter_definition(
            inputs,
            template,
        ),
        position_count=position_count,
    )
    if all(
        state is None
        for state in grid.states
    ):
        raise WorkflowError(
            "Die Mechanik kann f\u00fcr keine "
            "Bremsstellung aufgebaut werden; "
            "bitte die Parameter pr\u00fcfen."
        )
    return _brake_position_report(
        grid,
        title=(
            "TraSim \u2014 Mischer-Vorschau "
            "(Startmechanismus)"
        ),
    )


def run_mixed_brake_optimization(
    inputs: MixedBrakeInputs,
    template: ParameterSet,
    settings: MixedBrakeSettings | None = None,
    progress=None,
) -> MixedBrakeResult:
    """Run the mixed brake optimization.

    ``progress(generation, best_score)`` is called once per
    generation so UIs can show live progress.

    Raises WorkflowError with a user-facing message when the
    optimization cannot produce a result.
    """
    if settings is None:
        settings = MixedBrakeSettings()
    grid = BrakeGridSettings(
        brake_min_deg=inputs.brake_min_deg,
        brake_max_deg=inputs.brake_max_deg,
        position_count=settings.brake_positions,
    )
    brake_positions = make_brake_grid(grid)
    motion = MotionRange(
        start_angle=math.radians(
            inputs.motion_start_deg
        ),
        max_angle=math.radians(
            inputs.motion_travel_deg
        ),
        step=math.radians(
            inputs.motion_step_deg
        ),
    )
    input_angles = tuple(angle for angle in motion)
    targets = blend_target_curves(
        inputs.unbraked,
        inputs.braked,
        grid.fractions,
        input_angles=input_angles,
    )
    builder = MixedBrakeBuilder(inputs.definition)
    simulator = MechanismSimulator(motion=motion)
    fitness = BrakeCurveFitness(
        weights=tuple(
            1.0 for _ in brake_positions
        ),
        max_weight=1.0,
        endpoint_constraint=EndpointConstraint(
            end_angle=0.0,
            weight=1.0,
        ),
    )
    constraint = TransmissionAngleConstraint(
        minimum_angle=math.radians(
            settings.transmission_angle_deg
        ),
        weight=settings.transmission_weight,
    )
    optimizer = MixedBrakeOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
        definition=inputs.definition,
        brake_positions=brake_positions,
        targets=targets,
        transmission_angle_constraint=constraint,
    )
    rng = random.Random(settings.seed)
    population = PopulationFactory(
        random_generator=rng,
        initial_spread=0.1,
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
            random_generator=rng,
        ),
        target_fitness=settings.target_fitness,
        max_generations=settings.max_generations,
        stagnation_limit=settings.stagnation_limit,
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
        children_count=settings.children_count,
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
    optimized_definition = (
        _parameter_definition(
            inputs,
            best,
        )
    )
    result_grid = build_mixed_brake_grid(
        inputs,
        optimized_definition,
        position_count=(
            settings.brake_positions
        ),
        motion=motion,
    )
    html = _brake_position_report(
        result_grid,
        title=(
            "TraSim \u2014 Mischer-Optimierung "
            "(beste Geometrie)"
        ),
    )
    min_angle_deg = (
        result_grid.min_transmission_angle_deg
    )
    with tempfile.TemporaryDirectory() as temp:
        mechanism_path = (
            Path(temp) / "mechanism.csv"
        )
        CsvWriter.write_mechanism(
            optimized_definition,
            mechanism_path,
        )
        mechanism_csv = mechanism_path.read_text(
            encoding="utf-8",
        )
    return MixedBrakeResult(
        best_score=engine.best_score,
        stop_reason=(
            engine.stop_reason or ""
        ),
        generations=generations,
        brake_positions_deg=(
            grid.brake_positions_deg
        ),
        fitness_history=tuple(history),
        best_values=best.values(),
        html=html,
        min_transmission_angle_deg=(
            min_angle_deg
        ),
        optimized_definition=(
            optimized_definition
        ),
        mechanism_csv=mechanism_csv,
    )
