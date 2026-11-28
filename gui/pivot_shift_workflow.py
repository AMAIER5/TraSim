"""gui/pivot_shift_workflow.py

End-to-end workflow for the PIVOT SHIFT (bremshebel-los) mode:
a fixed-pivot mechanism CSV whose lever pair shares one common
pivot (the shifted brake state), two target curves (unbraked and
fully braked) and the evolutionary optimization over the two
endpoints of the pivot displacement:

(1) mutation of all parameters (incl. the pair pivot's
    shift dx/dy/dz, bounded by the optional per-axis pivot
    limits in the mechanism definition),
(2) frozen coupling rod lengths in the unbraked position,
(3) unbraked transfer function,
(4) braked transfer function with IDENTICAL coupling rods,
(5) fitness from both transfer functions.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from analysis.brake_curve_fitness import (
    BrakeCurveFitness,
)
from analysis.curve_plotter import (
    BoxPlot,
    CurvePlotter,
    mechanism_states_from_results,
)
from analysis.target_curve import TargetCurve
from analysis.transmission_angle import (
    MIN_TRANSMISSION_ANGLE,
    TransmissionAngleConstraint,
    min_transmission_angle,
)
from core.point3d import Point3D
from gui.workflow import (
    OptimizationStopped,
    WorkflowError,
    _text_to_temp_file,
)
from mechanics.pivot_shift_builder import (
    MechanismBuildError,
    PivotShiftBuilder,
)
from mechanism_io.csv_reader import CsvReader
from mechanism_io.csv_writer import CsvWriter
from model.mechanism_definition import (
    MechanismDefinition,
)
from optimization.csv_parameter_factory import (
    CsvParameterFactory,
    DEFAULT_SHIFT_OFFSET,
)
from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.parameter_mutation import (
    ParameterMutation,
)
from optimization.parameter_set import ParameterSet
from optimization.pivot_shift_optimizer import (
    PivotShiftOptimizer,
)
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import Reproduction
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange
from simulation.point_motion import PointMotion
from simulation.simulation_result import (
    SimulationResult,
)

__all__ = [
    "PivotShiftInputs",
    "PivotShiftResult",
    "PivotShiftSettings",
    "load_pivot_shift_inputs",
    "pivot_pair_candidates",
    "pivot_shift_inputs_from_parts",
    "pivot_shift_parameter_template",
    "pivot_shift_result_report",
    "run_pivot_shift_optimization",
    "shift_room_box",
]


@dataclass
class PivotShiftSettings:
    """Settings of the pivot shift optimization."""

    population_size: int = 40
    children_count: int = 40
    selection_count: int = 12
    mutation_strength: float = 0.03
    max_generations: int = 40
    stagnation_limit: int = 20
    stagnation_tolerance: float = 1e-8
    target_fitness: float = 0.005
    transmission_angle_deg: float = (
        math.degrees(MIN_TRANSMISSION_ANGLE)
    )
    transmission_weight: float = 1.0
    shift_effect_weight: float = 1.0
    gradient_weight: float = 0.0
    workers: int = 0
    seed: int = 42


@dataclass(frozen=True, slots=True)
class PivotShiftInputs:
    """Parsed and validated pivot shift inputs."""

    definition: MechanismDefinition
    unbraked: TargetCurve
    braked: TargetCurve
    shift_lever_id: int
    motion_start_deg: float
    motion_travel_deg: float
    motion_step_deg: float
    input_angles: (
        tuple[float, ...] | None
    ) = None

    """``input_angles`` is the motion support grid (radians)
    taken from the support points of the unbraked target curve
    (GUI pivot shift mode); ``None`` falls back to the evenly
    spaced ``MotionRange`` built from the ``motion_*_deg``
    fields."""


@dataclass(frozen=True, slots=True)
class PivotShiftResult:
    """Outcome of a completed pivot shift optimization."""

    best_score: float
    stop_reason: str
    generations: int
    fitness_history: tuple[float, ...]
    best_values: dict[str, float]
    input_angles_deg: tuple[float, ...]
    unbraked_target_deg: tuple[float, ...]
    unbraked_actual_deg: tuple[float, ...]
    braked_target_deg: tuple[float, ...]
    braked_actual_deg: tuple[float, ...]
    braked_buildable: bool
    shift_mm: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    min_transmission_angle_deg: float = field(
        default=0.0,
    )
    optimized_definition: (
        MechanismDefinition | None
    ) = None
    mechanism_csv: str = ""
    html: str = ""


def pivot_pair_candidates(
    definition: MechanismDefinition,
) -> list[frozenset[int]]:
    """All lever groups sharing a common pivot (the
    potential shift pairs of a mechanism)."""
    groups: dict[
        tuple[float, float, float],
        list[int],
    ] = {}
    for lever in definition.levers:
        groups.setdefault(
            (
                lever.pivot.x,
                lever.pivot.y,
                lever.pivot.z,
            ),
            [],
        ).append(lever.id)
    return [
        frozenset(ids)
        for ids in groups.values()
        if len(ids) > 1
    ]


def _validate_definition(
    definition: MechanismDefinition,
    shift_lever_id: int | None,
) -> int:
    if any(
        lever.pivot_on is not None
        for lever in definition.levers
    ):
        raise WorkflowError(
            "Der Pivot-Shift-Modus benötigt Hebel "
            "mit festen Drehpunkten; ein Hebel mit "
            "pivot_on (Bremshebel) gehört in den "
            "Mischer-Modus.",
        )
    candidates = pivot_pair_candidates(
        definition,
    )
    if not candidates:
        raise WorkflowError(
            "Für den Pivot-Shift-Modus müssen zwei "
            "Hebel einen gemeinsamen Drehpunkt "
            "besitzen (Hebelpaar).",
        )
    if shift_lever_id is None:
        if len(candidates) != 1:
            raise WorkflowError(
                "Mehrere Hebelpaare teilen sich "
                "Drehpunkte; bitte den Shift-Hebel "
                "auswählen.",
            )
        shift_lever_id = min(candidates[0])
    else:
        definition.get_lever(shift_lever_id)
        if not any(
            shift_lever_id in candidate
            for candidate in candidates
        ):
            raise WorkflowError(
                f"Hebel {shift_lever_id} teilt seinen "
                "Drehpunkt mit keinem anderen Hebel.",
            )
    return shift_lever_id


def pivot_shift_inputs_from_parts(
    definition: MechanismDefinition,
    unbraked: TargetCurve,
    braked: TargetCurve,
    *,
    shift_lever_id: int | None = None,
    motion_start_deg: float = -10.0,
    motion_travel_deg: float = 20.0,
    motion_step_deg: float = 2.0,
    input_angles: (
        tuple[float, ...] | None
    ) = None,
) -> PivotShiftInputs:
    """Assemble and validate the pivot shift inputs from
    already-parsed parts (GUI workflow)."""
    try:
        resolved = _validate_definition(
            definition,
            shift_lever_id,
        )
    except Exception as error:
        if isinstance(error, WorkflowError):
            raise
        raise WorkflowError(
            f"Ungültige Mechanismus-Definition: "
            f"{error}",
        ) from error
    return PivotShiftInputs(
        definition=definition,
        unbraked=unbraked,
        braked=braked,
        shift_lever_id=resolved,
        motion_start_deg=motion_start_deg,
        motion_travel_deg=motion_travel_deg,
        motion_step_deg=motion_step_deg,
        input_angles=input_angles,
    )


def load_pivot_shift_inputs(
    mechanism_csv: str,
    unbraked_csv: str,
    braked_csv: str,
    *,
    shift_lever_id: int | None = None,
    motion_start_deg: float = -10.0,
    motion_travel_deg: float = 20.0,
    motion_step_deg: float = 2.0,
) -> PivotShiftInputs:
    """Load and validate the pivot shift inputs from CSV
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
            definition = CsvReader.read_mechanism(
                mechanism_path,
            )
            unbraked = (
                TargetCurve.from_csv_strict(
                    unbraked_path,
                )
            )
            braked = (
                TargetCurve.from_csv_strict(
                    braked_path,
                )
            )
            angles = unbraked.input_angles
            return (
                pivot_shift_inputs_from_parts(
                    definition,
                    unbraked,
                    braked,
                    shift_lever_id=shift_lever_id,
                    motion_start_deg=(
                        motion_start_deg
                    ),
                    motion_travel_deg=(
                        motion_travel_deg
                    ),
                    motion_step_deg=(
                        motion_step_deg
                    ),
                    input_angles=tuple(
                        angles,
                    ),
                )
            )
    except WorkflowError:
        raise
    except Exception as error:
        raise WorkflowError(
            f"Eingabedatei konnte nicht gelesen "
            f"werden: {error}",
        ) from error


def pivot_shift_parameter_template(
    inputs: PivotShiftInputs,
) -> ParameterSet:
    """Optimization parameter template of the pivot shift
    mechanism (standard parameters plus the pair pivot's
    shift room)."""
    return CsvParameterFactory.create_pivot_shift(
        inputs.definition,
        shift_lever_id=inputs.shift_lever_id,
    )


def _middle_support_index(
    angles: tuple[float, ...],
) -> int:
    """Zero-based index of the middle support position
    (same round-half-up convention as the mixed brake
    workflow)."""
    return int(math.floor(len(angles) / 2 + 0.5)) - 1


def _reference_output_angle(
    inputs: PivotShiftInputs,
) -> float:
    """Raw target output of the unbraked curve at the
    middle support position (radians).  This is the
    anchor the output lever's start angle is mapped
    onto (same convention as the mixed brake mode);
    the braked curve stays in the same raw frame, so
    the data-side difference between the curves is
    preserved."""
    support = inputs.input_angles
    if support is None:
        support = tuple(
            angle for angle in _motion_range(inputs)
        )
    return inputs.unbraked.evaluate(
        support[_middle_support_index(support)]
    )


def _motion_range(
    inputs: PivotShiftInputs,
) -> MotionRange:
    return MotionRange(
        start_angle=math.radians(
            inputs.motion_start_deg,
        ),
        max_angle=math.radians(
            inputs.motion_travel_deg,
        ),
        step=math.radians(
            inputs.motion_step_deg,
        ),
    )


def _motion(inputs: PivotShiftInputs):
    if inputs.input_angles:
        return PointMotion(
            angles=tuple(inputs.input_angles),
        )
    return _motion_range(inputs)


def _evaluator(
    inputs: PivotShiftInputs,
    settings: PivotShiftSettings,
) -> Callable[[ParameterSet], float]:
    builder = PivotShiftBuilder(
        inputs.definition,
        shift_lever_id=inputs.shift_lever_id,
    )
    fitness = BrakeCurveFitness(
        weights=(1.0, 1.0),
        max_weight=1.0,
        gradient_weight=(settings.gradient_weight),
    )
    constraint = TransmissionAngleConstraint(
        minimum_angle=math.radians(
            settings.transmission_angle_deg,
        ),
        weight=settings.transmission_weight,
    )
    simulator = MechanismSimulator(
        motion=_motion(inputs),
    )
    optimizer = PivotShiftOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
        definition=inputs.definition,
        targets=(inputs.unbraked, inputs.braked),
        shift_lever_id=inputs.shift_lever_id,
        transmission_angle_constraint=constraint,
        shift_effect_weight=(
            settings.shift_effect_weight
        ),
        anchor_output=True,
        reference_output_angle=(
            _reference_output_angle(inputs)
        ),
    )
    return optimizer.evaluate


def _reduce_shift_to_buildable(
    optimizer: PivotShiftOptimizer,
    parameters: ParameterSet,
) -> tuple[float, float, float]:
    """Shift of the best candidate, reduced along its own
    direction to the largest displacement whose braked
    mechanism still builds and simulates without blocking
    ((0, 0, 0) when nothing works)."""
    shift = optimizer.shift_for(parameters)
    if shift == (0.0, 0.0, 0.0):
        return shift
    for fraction in (
        1.0,
        0.9,
        0.75,
        0.5,
        0.25,
    ):
        candidate = tuple(
            component * fraction
            for component in shift
        )
        result = optimizer._cached_simulation(
            parameters,
            candidate,
        )
        if (
            result is not None
            and result.success
        ):
            return candidate
    return (0.0, 0.0, 0.0)


def _combined_curve(
    results: tuple[SimulationResult, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Combine per-stage simulations into one transfer
    curve: the stage-0 input angles paired with the final
    stage's output angles, trimmed to the shorter length
    (same convention as the optimizers)."""
    if not results:
        return ((), ())
    count = min(
        len(results[0].input_angles),
        len(results[-1].output_angles),
    )
    return (
        results[0].input_angles[:count],
        results[-1].output_angles[:count],
    )


def shift_room_box(
    definition: MechanismDefinition,
    shift_lever_id: int,
) -> BoxPlot:
    """The room within which the pair pivot may be
    displaced for the braked state (issue #140), as a
    wireframe box around the shift lever's pivot.

    The per-axis bounds come from the shift lever's
    optional pivot limits on every axis (x, y and z);
    unrestricted axes fall back to the symmetric default
    shift offset (same room the optimizer uses)."""
    lever = definition.get_lever(shift_lever_id)
    (
        (x_min, x_max),
        (y_min, y_max),
        (z_min, z_max),
    ) = lever.pivot_limits
    x0 = (
        x_min
        if x_min is not None
        else lever.pivot.x - DEFAULT_SHIFT_OFFSET
    )
    x1 = (
        x_max
        if x_max is not None
        else lever.pivot.x + DEFAULT_SHIFT_OFFSET
    )
    y0 = (
        y_min
        if y_min is not None
        else lever.pivot.y - DEFAULT_SHIFT_OFFSET
    )
    y1 = (
        y_max
        if y_max is not None
        else lever.pivot.y + DEFAULT_SHIFT_OFFSET
    )
    z0 = (
        z_min
        if z_min is not None
        else lever.pivot.z - DEFAULT_SHIFT_OFFSET
    )
    z1 = (
        z_max
        if z_max is not None
        else lever.pivot.z + DEFAULT_SHIFT_OFFSET
    )
    return BoxPlot(
        name="Verschieberaum Drehpunkt",
        minimum=Point3D(x=x0, y=y0, z=z0),
        maximum=Point3D(x=x1, y=y1, z=z1),
    )


def pivot_shift_result_report(
    *,
    input_angles_deg: tuple[float, ...],
    unbraked_target_deg: tuple[float, ...],
    unbraked_actual_deg: tuple[float, ...],
    braked_target_deg: tuple[float, ...],
    braked_actual_deg: tuple[float, ...],
    unbraked_states: tuple = (),
    braked_states: tuple = (),
    box: BoxPlot | None = None,
    title: str = (
        "TraSim \u2014 Ergebnis "
        "(Pivot-Shift-Optimierung, "
        "optimierte Geometrie)"
    ),
) -> str:
    """HTML result report of a pivot shift run (issue
    #140): the mechanism of the optimized geometry in
    the unbraked AND the braked position under the
    achieved target curves, plus the shift room of the
    pair pivot drawn as a box.

    The braked snapshots are drawn with their own
    legend groups (suffixed names) so both states stay
    separately toggleable in the 3D diagram."""
    plotter = CurvePlotter(title=title)
    input_rad = tuple(
        math.radians(angle)
        for angle in input_angles_deg
    )
    plotter.add_target_curve(
        input_rad,
        tuple(
            math.radians(angle)
            for angle in unbraked_target_deg
        ),
        label="Soll ungebremst",
    )
    plotter.add_target_curve(
        input_rad,
        tuple(
            math.radians(angle)
            for angle in braked_target_deg
        ),
        label="Soll voll gebremst",
    )
    if unbraked_actual_deg:
        plotter.add_actual_curve(
            input_rad,
            tuple(
                math.radians(angle)
                for angle in unbraked_actual_deg
            ),
            label="Ist ungebremst",
        )
    if braked_actual_deg:
        plotter.add_actual_curve(
            input_rad,
            tuple(
                math.radians(angle)
                for angle in braked_actual_deg
            ),
            label="Ist voll gebremst",
        )
    for state in unbraked_states:
        plotter.add_mechanism_state(state)
    for state in braked_states:
        plotter.add_mechanism_state(state)
    if box is not None:
        plotter.add_box(box)
    return plotter.build_html()


def run_pivot_shift_optimization(
    inputs: PivotShiftInputs,
    template: ParameterSet,
    settings: PivotShiftSettings | None = None,
    progress=None,
    on_best: (
        Callable[[int, dict[str, float]], None]
        | None
    ) = None,
) -> PivotShiftResult:
    """Run the pivot shift optimization.

    ``progress(generation, best_score)`` is called once
    per generation; ``on_best(generation, values)`` is
    called whenever the best candidate improves (UIs use
    it to persist intermediate results).

    Raises WorkflowError with a user-facing message when
    the optimization cannot produce a result.
    """
    if settings is None:
        settings = PivotShiftSettings()
    # The motion support grid follows the unbraked target
    # curve's support points when available (GUI), so the
    # simulated curves are sampled exactly where the
    # targets are defined.
    motion = _motion(inputs)
    simulator = MechanismSimulator(motion=motion)
    support = tuple(motion)
    builder = PivotShiftBuilder(
        inputs.definition,
        shift_lever_id=inputs.shift_lever_id,
    )
    fitness = BrakeCurveFitness(
        weights=(1.0, 1.0),
        max_weight=1.0,
        gradient_weight=(settings.gradient_weight),
    )
    constraint = TransmissionAngleConstraint(
        minimum_angle=math.radians(
            settings.transmission_angle_deg,
        ),
        weight=settings.transmission_weight,
    )
    optimizer = PivotShiftOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
        definition=inputs.definition,
        targets=(inputs.unbraked, inputs.braked),
        shift_lever_id=inputs.shift_lever_id,
        transmission_angle_constraint=constraint,
        shift_effect_weight=(
            settings.shift_effect_weight
        ),
        anchor_output=True,
        reference_output_angle=(
            _reference_output_angle(inputs)
        ),
    )

    worker_count = settings.workers
    if worker_count <= 0:
        worker_count = multiprocessing.cpu_count()
    worker_count = max(1, worker_count)

    evaluator_factory = partial(
        _evaluator,
        inputs,
        settings,
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
                strength=(settings.mutation_strength),
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
    engine.update_best()
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
            and engine.best_score < last_best_score
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
        optimizer.apply_parameters(best)
    )
    shift = _reduce_shift_to_buildable(
        optimizer,
        best,
    )
    # Result curves in SHAFT coordinates: simulate the final
    # geometry with the best candidate's drive shaft offset
    # applied (the drive lever runs in its own lever-angle
    # frame while the reported input axis stays the virtual
    # shaft of the target curves).
    shaft_offset = optimizer.drive_shaft_offset(
        best,
    )
    result_motion = PointMotion(
        angles=tuple(
            angle + shaft_offset
            for angle in support
        ),
    )
    result_simulator = MechanismSimulator(
        motion=result_motion,
    )

    result_builder = PivotShiftBuilder(
        optimized_definition,
        shift_lever_id=inputs.shift_lever_id,
    )
    # Issue #141: the reported actual curves follow the
    # optimizer's output anchoring (same convention as the
    # mixed brake mode):  the candidate's output-lever start
    # angle is mapped onto the raw target output of the
    # unbraked curve at the middle support position, so the
    # Ist curves are directly comparable with the Soll
    # curves instead of carrying the constant installation
    # offset of the output lever.
    output_offset = optimizer.output_offset(
        best,
    )
    unbraked_actual: tuple[float, ...] = ()
    braked_actual: tuple[float, ...] = ()
    braked_buildable = False
    mechanisms = []
    unbraked_results = None
    braked_results = None
    try:
        mechanism = result_builder.build(
            (0.0, 0.0, 0.0),
        )
        mechanisms.append(mechanism)
        unbraked_results = (
            result_simulator.simulate(mechanism)
        )
        _angles, outputs = _combined_curve(
            unbraked_results,
        )
        if outputs:
            unbraked_actual = tuple(
                math.degrees(
                    value - output_offset
                )
                for value in outputs
            )
    except MechanismBuildError:
        pass
    try:
        mechanism = result_builder.build(shift)
        braked_results = (
            result_simulator.simulate(mechanism)
        )
        _, outputs = _combined_curve(
            braked_results,
        )
        if outputs:
            braked_actual = tuple(
                math.degrees(
                    value - output_offset
                )
                for value in outputs
            )
            mechanisms.append(mechanism)
            braked_buildable = True
    except MechanismBuildError:
        pass

    minimum_deg = math.degrees(
        MIN_TRANSMISSION_ANGLE,
    )
    if mechanisms:
        minimum = min(
            min_transmission_angle(mechanism)
            for mechanism in mechanisms
        )
        minimum_deg = math.degrees(minimum)

    # Issue #140: mechanism snapshots of the optimized
    # geometry in the unbraked AND the braked position,
    # drawn under the achieved curves together with the
    # shift room of the pair pivot.
    unbraked_states = ()
    if (
        mechanisms
        and unbraked_results is not None
    ):
        unbraked_states = (
            mechanism_states_from_results(
                mechanisms[0],
                unbraked_results,
            )
        )
    braked_states = ()
    if (
        braked_buildable
        and braked_results is not None
    ):
        braked_lever_names = {
            lever.id: (
                f"Lever {lever.id} gebremst"
            )
            for lever in optimized_definition.levers
        }
        braked_lever_colors = {
            lever.id: "#64748b"
            for lever in optimized_definition.levers
        }
        braked_states = (
            mechanism_states_from_results(
                mechanisms[-1],
                braked_results,
                lever_names=(
                    braked_lever_names
                ),
                lever_colors=(
                    braked_lever_colors
                ),
            )
        )
    input_angles_deg = tuple(
        math.degrees(angle)
        for angle in support
    )
    unbraked_target_deg = tuple(
        math.degrees(value)
        for value in inputs.unbraked.sample(
            support,
        ).output_angles
    )
    braked_target_deg = tuple(
        math.degrees(value)
        for value in inputs.braked.sample(
            support,
        ).output_angles
    )
    result_html = pivot_shift_result_report(
        input_angles_deg=input_angles_deg,
        unbraked_target_deg=(
            unbraked_target_deg
        ),
        unbraked_actual_deg=(
            unbraked_actual
        ),
        braked_target_deg=braked_target_deg,
        braked_actual_deg=braked_actual,
        unbraked_states=unbraked_states,
        braked_states=braked_states,
        box=shift_room_box(
            optimized_definition,
            inputs.shift_lever_id,
        ),
    )

    with tempfile.TemporaryDirectory() as temp:
        mechanism_path = Path(temp) / "mechanism.csv"
        CsvWriter.write_mechanism(
            optimized_definition,
            mechanism_path,
        )
        optimized_csv = mechanism_path.read_text(
            encoding="utf-8",
        )

    return PivotShiftResult(
        best_score=engine.best_score,
        stop_reason=(engine.stop_reason or ""),
        generations=generations,
        fitness_history=tuple(history),
        best_values=best.values(),
        input_angles_deg=input_angles_deg,
        unbraked_target_deg=(
            unbraked_target_deg
        ),
        unbraked_actual_deg=unbraked_actual,
        braked_target_deg=braked_target_deg,
        braked_actual_deg=braked_actual,
        braked_buildable=braked_buildable,
        shift_mm=shift,
        min_transmission_angle_deg=minimum_deg,
        optimized_definition=optimized_definition,
        mechanism_csv=optimized_csv,
        html=result_html,
    )
