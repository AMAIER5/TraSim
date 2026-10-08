"""optimization/mixed_brake_optimizer.py

Optimizer adapter for brake-curve optimization with the
MixedBrakeBuilder.

Each candidate (ParameterSet) is evaluated over all brake
positions (lever angles ``beta_k`` of the brake lever):

1. The mechanism is rebuilt for every ``beta_k`` via
   ``MixedBrakeBuilder.build(beta_k)``.
2. The complete rudder curve is simulated for every build
   with the shared ``MechanismSimulator`` (last stage =
   rudder stage).
3. All (SimulationResult, TargetCurve) pairs are scored with
   the multi-curve ``BrakeCurveFitness`` from S3.

A ``MechanismBuildError`` never crashes the optimization:
the candidate receives a hard penalty value instead
(``PENALTY_MAX_BLOCKING``).

Simulation results are cached per (candidate, beta) pair, so
re-evaluating the same candidate does not rebuild and
re-simulate the mechanisms.
"""

from __future__ import annotations

import logging

from analysis.brake_curve_fitness import (
    PENALTY_MAX_BLOCKING,
    BrakeCurveFitness,
)
from analysis.target_curve import TargetCurve
from analysis.transmission_angle import (
    TransmissionAngleConstraint,
)
from core.point3d import Point3D
from mechanics.mechanism import Mechanism
from mechanics.mixed_brake_builder import (
    MechanismBuildError,
    MixedBrakeBuilder,
)
from model.mechanism_definition import MechanismDefinition
from optimization.parameter_set import ParameterSet
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.simulation_result import SimulationResult

logger = logging.getLogger(__name__)

__all__ = [
    "PENALTY_MAX_BLOCKING",
    "MixedBrakeOptimizer",
]


class MixedBrakeOptimizer:
    """Evaluates candidates over multiple brake positions.

    Parameters
    ----------
    builder:
        The (parameterless) MixedBrakeBuilder used to build
        one mechanism per brake position.  A builder is
        created per candidate definition internally, so a
        single shared builder instance may be passed.
    simulator:
        Simulates the rudder curve for each built
        mechanism.  The last simulated stage is used as
        the rudder stage.
    fitness:
        BrakeCurveFitness scoring all (result, target)
        pairs.
    definition:
        Base mechanism definition; candidate parameters are
        applied as an overlay (length, angle, pivot).
    brake_positions:
        Lever angles (radians) of the brake lever, ordered
        by increasing brake position.
    targets:
        One TargetCurve per brake position.
    """

    def __init__(
        self,
        *,
        builder: MixedBrakeBuilder,
        simulator: MechanismSimulator,
        fitness: BrakeCurveFitness,
        definition: MechanismDefinition,
        brake_positions: tuple[float, ...],
        targets: tuple[TargetCurve, ...],
        transmission_angle_constraint: (
            TransmissionAngleConstraint | None
        ) = None,
    ) -> None:
        if len(brake_positions) == 0:
            raise ValueError(
                "At least one brake position is required."
            )
        if len(brake_positions) != len(targets):
            raise ValueError(
                "Number of targets must match number of "
                "brake positions."
            )
        self._builder = builder
        self._simulator = simulator
        self._fitness = fitness
        self._definition = definition
        self._brake_positions = tuple(brake_positions)
        self._targets = tuple(targets)
        self._transmission_constraint = (
            transmission_angle_constraint
        )
        self._cache: dict[
            tuple[ParameterSet, float],
            SimulationResult,
        ] = {}
        self._mechanism_cache: dict[
            tuple[ParameterSet, float],
            Mechanism,
        ] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._evaluations = 0
        self._build_errors = 0

    def evaluate(
        self,
        parameters: ParameterSet,
    ) -> float:
        """Evaluate one candidate over all brake positions.

        Returns the fitness (lower is better).  Candidates
        that cannot be built for at least one brake position
        receive a hard penalty instead of raising.
        """
        self._evaluations += 1
        pairs = []
        mechanisms = []
        for beta, target in zip(
            self._brake_positions,
            self._targets,
        ):
            result = self._cached_simulation(
                parameters,
                beta,
            )
            if result is None:
                logger.debug(
                    "Build error for candidate %s at "
                    "beta=%s; penalty applied.",
                    parameters,
                    beta,
                )
                return PENALTY_MAX_BLOCKING
            pairs.append((result, target))
            if self._transmission_constraint is not None:
                mechanism = self._cached_mechanism(
                    parameters,
                    beta,
                )
                if mechanism is not None:
                    mechanisms.append(mechanism)
        fitness = self._fitness.evaluate(pairs)
        if self._transmission_constraint is not None:
            fitness += (
                self._transmission_constraint.evaluate(
                    mechanisms
                )
            )
        return fitness

    def _cached_mechanism(
        self,
        parameters: ParameterSet,
        beta: float,
    ) -> Mechanism | None:
        """Return the cached built mechanism for
        (candidate, beta) or build it.  None signals a
        build error.
        """
        key = (parameters, beta)
        cached = self._mechanism_cache.get(key)
        if cached is not None:
            return cached
        try:
            builder = self._builder_for(parameters)
            mechanism = builder.build(
                self._clamp_mixer_angle(beta)
            )
        except MechanismBuildError:
            return None
        self._mechanism_cache[key] = mechanism
        return mechanism

    def _cached_simulation(
        self,
        parameters: ParameterSet,
        beta: float,
    ) -> SimulationResult | None:
        """Return the cached simulation for (candidate, beta)
        or compute it.  None signals a build error (the
        caller applies a penalty).
        """
        key = (parameters, beta)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        result = self._simulate(
            parameters,
            beta,
        )
        if result is not None:
            self._cache[key] = result
        return result

    def _simulate(
        self,
        parameters: ParameterSet,
        beta: float,
    ) -> SimulationResult | None:
        """Build and simulate one (candidate, beta) pair.

        Returns None on MechanismBuildError.
        """
        try:
            builder = self._builder_for(parameters)
            mechanism = builder.build(
                self._clamp_mixer_angle(beta)
            )
        except MechanismBuildError:
            self._build_errors += 1
            return None
        results = self._simulator.simulate(mechanism)
        if len(results) == 0:
            return None
        return results[-1]

    def _builder_for(
        self,
        parameters: ParameterSet,
    ) -> MixedBrakeBuilder:
        """Create a builder for the candidate's parameter
        overlay; falls back to the shared builder when the
        candidate matches the base definition.
        """
        definition = self._apply_parameters(parameters)
        if definition == self._definition:
            return self._builder
        return MixedBrakeBuilder(definition)

    def _apply_parameters(
        self,
        parameters: ParameterSet,
    ) -> MechanismDefinition:
        """Apply optimization parameters (length, angle,
        pivot x/y) onto the base definition.
        """
        from dataclasses import replace

        values = parameters.values()
        levers = []
        for lever in self._definition.levers:
            length = values.get(
                f"lever.{lever.id}.length",
                lever.length_start,
            )
            angle = self._clamp(
                values.get(
                    f"lever.{lever.id}.angle",
                    lever.angle_start,
                ),
                lever.angle_min,
                lever.angle_max,
            )
            pivot_x = values.get(
                f"lever.{lever.id}.pivot.x",
                lever.pivot.x,
            )
            pivot_y = values.get(
                f"lever.{lever.id}.pivot.y",
                lever.pivot.y,
            )
            lever = replace(
                lever,
                length_start=length,
                angle_start=angle,
                pivot=Point3D(
                    x=pivot_x,
                    y=pivot_y,
                    z=lever.pivot.z,
                ),
            )
            levers.append(lever)
        return MechanismDefinition(tuple(levers))

    def _clamp_mixer_angle(
        self,
        beta: float,
    ) -> float:
        """Clamp the mixer angle to the brake lever's
        bounds (behaviour matches existing levers, whose
        reference angles are clamped instead of raising).
        """
        brake = self._definition.get_lever(
            self._brake_lever_id
        )
        return self._clamp(
            beta,
            brake.angle_min,
            brake.angle_max,
        )

    @property
    def _brake_lever_id(self) -> int:
        mixer = next(
            lever
            for lever in self._definition.levers
            if lever.pivot_on is not None
        )
        return mixer.pivot_reference.lever_id

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(minimum, min(maximum, value))

    def clear_cache(self) -> None:
        """Remove all cached simulations."""
        self._cache.clear()
        self._mechanism_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._evaluations = 0
        self._build_errors = 0

    def get_cache_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "evaluations": self._evaluations,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "build_errors": self._build_errors,
        }
