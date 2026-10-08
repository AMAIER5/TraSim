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
from gui.workflow import (
    WorkflowError,
    _text_to_temp_file,
)
from mechanics.mechanism import Mechanism
from mechanics.mixed_brake_builder import (
    MixedBrakeBuilder,
)
from mechanism_io.csv_reader import CsvReader
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
from simulation.simulation_result import (
    SimulationResult,
)
from simulation.motion_range import MotionRange

__all__ = [
    "MixedBrakeInputs",
    "MixedBrakeResult",
    "MixedBrakeSettings",
    "load_mixed_brake_inputs",
    "mixed_brake_parameter_template",
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
        brake_min_deg=(
            mixer.pivot_reference.angle_deg
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


def _simulate_best(
    optimizer: MixedBrakeOptimizer,
    best: ParameterSet,
    grid: BrakeGridSettings,
    *,
    targets: tuple[TargetCurve, ...],
    input_angles: tuple[float, ...],
) -> tuple[str, float]:
    """Build the HTML report of the best candidate over all
    brake positions and determine its minimum transmission
    angle.

    For every brake position the blended target curve
    (Soll) and the achieved output curve (Ist) are drawn,
    together with three mechanism snapshots of the middle
    brake position.
    """
    plotter = CurvePlotter(
        title=(
            "TraSim — Mischer-Optimierung "
            "(beste Geometrie)"
        ),
    )
    mechanisms: list[Mechanism] = []
    middle_mechanism: Mechanism | None = None
    middle_results: (
        tuple[SimulationResult, ...] | None
    ) = None
    for index, beta_deg in enumerate(
        grid.brake_positions_deg
    ):
        beta = math.radians(beta_deg)
        results = optimizer.simulate_stages(
            best,
            beta,
        )
        if results is None:
            continue
        mechanism = optimizer._cached_mechanism(
            best,
            beta,
        )
        if mechanism is None:
            continue
        mechanisms.append(mechanism)
        if index == len(grid.fractions) // 2:
            middle_mechanism = mechanism
            middle_results = results
        fraction = grid.fractions[index]
        plotter.add_target_curve(
            input_angles,
            targets[index].sample(
                input_angles,
            ).output_angles,
            label=(
                f"Soll Bremse {100 * fraction:.0f} %"
            ),
        )
        plotter.add_actual_curve(
            results[-1].input_angles,
            results[-1].output_angles,
            label=(
                f"Bremse {100 * fraction:.0f} %"
            ),
        )
    if (
        middle_mechanism is not None
        and middle_results is not None
    ):
        for state in mechanism_states_from_results(
            middle_mechanism,
            middle_results,
        ):
            plotter.add_mechanism_state(state)
    if not mechanisms:
        return (
            plotter.build_html(),
            float("inf"),
        )
    minimum = min(
        min_transmission_angle(mechanism)
        for mechanism in mechanisms
    )
    return (
        plotter.build_html(),
        math.degrees(minimum),
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
    html, min_angle_deg = _simulate_best(
        optimizer,
        best,
        grid,
        targets=targets,
        input_angles=input_angles,
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
    )
