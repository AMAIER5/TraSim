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
import multiprocessing
import random
import tempfile
from dataclasses import (
    dataclass,
    field,
)
from functools import partial
from pathlib import Path
from collections.abc import Callable

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
from analysis.transmission_angle import (
    TransmissionAngleConstraint,
    min_transmission_angle,
)
from core.point3d import Point3D
from gui.workflow import (
    OptimizationStopped,
    WorkflowError,
    _drive_lever_id,
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
from simulation.motion_provider import (
    MotionProvider,
)
from simulation.motion_range import MotionRange
from simulation.point_motion import PointMotion

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
    gradient_weight: float = 0.0
    workers: int = 0
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
    input_angles: (
        tuple[float, ...] | None
    ) = None

    """
    ``input_angles`` is the motion support grid (radians)
    taken from the support points of the unbraked target
    curve when that curve is discrete (GUI mixed brake
    mode).  When ``None`` (e.g. after
    ``load_mixed_brake_inputs`` with an interpolating
    curve), the motion falls back to the evenly spaced
    ``MotionRange`` built from the ``motion_*_deg`` fields.
    """


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
    support_angles = getattr(
        unbraked,
        "input_angles",
        None,
    )
    return MixedBrakeInputs(
        definition=definition,
        unbraked=unbraked,
        braked=braked,
        motion_start_deg=motion_start_deg,
        motion_travel_deg=motion_travel_deg,
        motion_step_deg=motion_step_deg,
        input_angles=(
            tuple(support_angles)
            if support_angles is not None
            else None
        ),
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
        unbraked = TargetCurve.from_csv_strict(
            unbraked_path,
        )
        braked = TargetCurve.from_csv_strict(
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
        input_angles=(
            unbraked.input_angles
        ),
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


def _motion(
    inputs: MixedBrakeInputs,
) -> MotionProvider:
    """Motion provider of the mixed brake workflow.

    When the (unbraked) target curve contributed its
    support points (``input_angles``), the mechanism is
    simulated exactly on those support points via
    ``PointMotion``.  This keeps the blended target
    curves consistent with a discrete target curve,
    which is only defined at its support points
    (issue #84: evaluating it on the fixed default
    -10..+10 deg grid raised a ValueError).

    The support angles are anchored at the drive
    lever's start angle, mirroring the normal
    workflow (``anchored_input_angles``): the drive
    lever sits at its start angle at the middle
    support position and every support point keeps
    its angular distance to that middle.

    Otherwise the evenly spaced ``MotionRange`` from
    the ``motion_*_deg`` fields is used as before.
    """
    if inputs.input_angles is None:
        return _motion_range(inputs)
    return PointMotion(
        angles=_anchored_support_angles(inputs),
    )


def _anchored_support_angles(
    inputs: MixedBrakeInputs,
) -> tuple[float, ...]:
    """Support-point input angles (radians) shifted so
    that the drive lever sits at its start angle at
    the middle support position of the target curve
    (same middle-position convention as the normal
    workflow)."""
    angles = inputs.input_angles
    middle = angles[
        int(math.floor(len(angles) / 2 + 0.5)) - 1
    ]
    drive_angle = inputs.definition.get_lever(
        _drive_lever_id(inputs.definition),
    ).angle_start
    return tuple(
        angle - middle + drive_angle
        for angle in angles
    )


def _blended_targets(
    inputs: MixedBrakeInputs,
    fractions: tuple[float, ...],
    input_angles: tuple[float, ...],
) -> tuple[TargetCurve, ...]:
    """One blended target curve per brake fraction,
    defined on the (anchored) motion angles.

    When the base curves are discrete (mixed brake
    GUI mode), they are evaluated on their own
    support points (issue #84: they cannot be
    evaluated on an arbitrary motion grid) and the
    blended outputs are then placed on the motion
    angles, mirroring the anchoring of the normal
    workflow.  The blended curves also blend the
    base curves' per-support-point weights the same
    way as their outputs (issue #87: the second
    target curve must also be weightable): the
    weight of a support point at brake fraction f is
    (1 - f) * unbraked weight + f * braked weight.
    Otherwise ``blend_target_curves`` evaluates the
    interpolating base curves directly on the
    motion angles as before.
    """
    support = inputs.input_angles
    if support is None:
        return blend_target_curves(
            inputs.unbraked,
            inputs.braked,
            fractions,
            input_angles=input_angles,
        )
    try:
        outputs = tuple(
            tuple(
                (1.0 - fraction)
                * inputs.unbraked.evaluate(angle)
                + fraction
                * inputs.braked.evaluate(angle)
                for angle in support
            )
            for fraction in fractions
        )
    except ValueError as error:
        raise WorkflowError(
            "Die Zielkurven m\u00fcssen auf denselben "
            "St\u00fctzstellen definiert sein: "
            f"{error}",
        ) from error
    unbraked_weights = getattr(
        inputs.unbraked,
        "weights",
        None,
    )
    braked_weights = getattr(
        inputs.braked,
        "weights",
        None,
    )
    if (
        unbraked_weights is not None
        and braked_weights is not None
    ):
        blended_weights = tuple(
            tuple(
                (1.0 - fraction) * unbraked
                + fraction * braked
                for unbraked, braked in zip(
                    unbraked_weights,
                    braked_weights,
                )
            )
            for fraction in fractions
        )
    else:
        blended_weights = tuple(
            None for _ in fractions
        )
    return tuple(
        TargetCurve.from_points_strict(
            input_angles=input_angles,
            output_angles=values,
            weights=weights,
        )
        for values, weights in zip(
            outputs,
            blended_weights,
        )
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
                        z=values.get(
                            f"lever.{lever.id}.pivot.z",
                            lever.pivot.z,
                        ),
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
    motion: MotionProvider | None = None,
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
        motion = _motion(inputs)
    input_angles = tuple(
        angle for angle in motion
    )
    targets = _blended_targets(
        inputs,
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
    motion: MotionProvider | None = None,
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
        input_rad = tuple(
            math.radians(angle)
            for angle in (
                state.input_angles_deg
            )
        )
        plotter.add_target_curve(
            input_rad,
            tuple(
                math.radians(angle)
                for angle in state.target_deg
            ),
            label=(
                f"Soll Bremse "
                f"{100 * fraction:.0f} %"
            ),
        )
        plotter.add_actual_curve(
            input_rad,
            tuple(
                math.radians(angle)
                for angle in state.actual_deg
            ),
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


def single_brake_position_report(
    title: str,
    state: MixedBrakeState,
    *,
    show_actual: bool = True,
) -> str:
    """HTML report of one brake position (issue #96:
    the degree values of the state are converted to
    radians for the plotter, so the diagram shows
    the usual degree axes)."""
    plotter = CurvePlotter(title=title)
    input_angles_rad = tuple(
        math.radians(angle)
        for angle in state.input_angles_deg
    )
    plotter.add_target_curve(
        input_angles_rad,
        tuple(
            math.radians(angle)
            for angle in state.target_deg
        ),
        label=(
            f"Soll Bremse "
            f"{100 * state.brake_fraction:.0f} %"
        ),
    )
    if show_actual:
        plotter.add_actual_curve(
            input_angles_rad,
            tuple(
                math.radians(angle)
                for angle in state.actual_deg
            ),
            label=(
                f"Bremse "
                f"{100 * state.brake_fraction:.0f} %"
            ),
        )
    for snapshot in state.states:
        plotter.add_mechanism_state(snapshot)
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


def _mixed_evaluator_factory(
    inputs: MixedBrakeInputs,
    settings: MixedBrakeSettings,
    brake_positions: tuple[float, ...],
    targets: tuple[TargetCurve, ...],
) -> Callable[[ParameterSet], float]:
    """Build a fresh mixed brake evaluator in the
    calling process.

    In parallel mode this runs once per worker
    process as the pool initializer's
    ``evaluator_factory`` (issue #86).  It must live
    at module level so Windows spawn can pickle it.
    """
    builder = MixedBrakeBuilder(
        inputs.definition,
    )
    motion = _motion(inputs)
    simulator = MechanismSimulator(
        motion=motion,
    )
    fitness = BrakeCurveFitness(
        weights=tuple(
            1.0 for _ in brake_positions
        ),
        max_weight=1.0,
        endpoint_constraint=EndpointConstraint(
            end_angle=0.0,
            weight=1.0,
        ),
        gradient_weight=(
            settings.gradient_weight
        ),
    )
    constraint = TransmissionAngleConstraint(
        minimum_angle=math.radians(
            settings.transmission_angle_deg,
        ),
        weight=settings.transmission_weight,
    )
    return MixedBrakeOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
        definition=inputs.definition,
        brake_positions=brake_positions,
        targets=targets,
        transmission_angle_constraint=(
            constraint
        ),
    ).evaluate


def run_mixed_brake_optimization(
    inputs: MixedBrakeInputs,
    template: ParameterSet,
    settings: MixedBrakeSettings | None = None,
    progress=None,
    on_best: (
        Callable[[int, dict[str, float]], None]
        | None
    ) = None,
) -> MixedBrakeResult:
    """Run the mixed brake optimization.

    ``progress(generation, best_score)`` is called once per
    generation so UIs can show live progress.
    ``on_best(generation, values)`` is called whenever the
    best candidate improves, with the generation number
    and the best candidate's parameter values (internal
    units); UIs use it to persist intermediate results so
    a user stop still yields the best values found so far
    (issue #93).

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
    motion = _motion(inputs)
    input_angles = tuple(angle for angle in motion)
    targets = _blended_targets(
        inputs,
        grid.fractions,
        input_angles=input_angles,
    )
    builder = MixedBrakeBuilder(
        inputs.definition,
    )
    simulator = MechanismSimulator(
        motion=motion,
    )

    worker_count = settings.workers
    if worker_count <= 0:
        worker_count = multiprocessing.cpu_count()
    worker_count = max(1, worker_count)

    evaluator_factory = partial(
        _mixed_evaluator_factory,
        inputs,
        settings,
        brake_positions,
        targets,
    )
    fitness = BrakeCurveFitness(
        weights=tuple(
            1.0 for _ in brake_positions
        ),
        max_weight=1.0,
        endpoint_constraint=EndpointConstraint(
            end_angle=0.0,
            weight=1.0,
        ),
        gradient_weight=(
            settings.gradient_weight
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
        evaluator=(
            optimizer.evaluate
            if worker_count == 1
            else None
        ),
        evaluator_factory=(
            evaluator_factory
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
        engine.close()
        raise WorkflowError(
            "Keine gültigen Startkandidaten — "
            "die Mechanik ist im Suchraum "
            "unzulässig. Bitte Parameterbereiche "
            "erweitern.",
        )
    history: list[float] = [engine.best_score]
    generations = 0
    last_best_score = engine.best_score
    for generation in engine.run(
        children_count=settings.children_count,
    ):
        generations += 1
        history.append(engine.best_score)
        if (
            on_best is not None
            and engine.best_score
            < last_best_score
        ):
            last_best_score = engine.best_score
            on_best(
                generations,
                engine.best_candidate.values(),
            )
        if progress is not None:
            try:
                progress(
                    generations,
                    engine.best_score,
                )
            except OptimizationStopped:
                engine.stop_reason = (
                    "stopped_by_user"
                )
                break
    engine.close()
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
