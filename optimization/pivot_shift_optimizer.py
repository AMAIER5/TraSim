"""optimization/pivot_shift_optimizer.py

Optimizer adapter for the PIVOT-SHIFT brake approach: the
brake state is the displacement of the common pivot of a
lever pair within a defined room (no modelled brake lever).

Every candidate (ParameterSet) is evaluated exactly at the
two endpoints of the displacement:

(1) All parameters are mutated by the evolution engine
    inside their allowed ranges (length, angle, pivot and the
    two shift parameters of the room).
(2) The coupling rod lengths are computed by the builder in
    the UNSHIFTED (unbraked) position and stay frozen.
(3) The transfer function of the unbraked position is
    simulated.
(4) The transfer function of the braked position is built and
    simulated with IDENTICAL coupling rods - braked and
    unbraked differ ONLY in the displaced pivot.
(5) The fitness value is computed from both transfer
    functions with BrakeCurveFitness.

A ``MechanismBuildError`` never crashes the evaluation: the
unbraked position is mandatory (its failure yields the hard
penalty ``PENALTY_MAX_BLOCKING``), while an unbuildable braked
position adds a penalty proportional to how much its target
curve differs from the unbraked one, so the optimizer is
driven to keep the braked position buildable (analogous to
the mixed brake optimizer's skip penalty).
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
from mechanics.mechanism import Mechanism
from mechanics.pivot_shift_builder import (
    MechanismBuildError,
    PivotShiftBuilder,
    resolve_pair_ids,
)
from model.mechanism_definition import (
    MechanismDefinition,
)
from optimization.parameter_set import (
    ParameterSet,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.simulation_result import (
    SimulationResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PENALTY_MAX_BLOCKING",
    "PivotShiftOptimizer",
    "shift_parameter_names",
]

#: Penalty factor for an unbuildable braked position
#: (proportional to the mean squared difference between the
#: braked and unbraked target curves).
BRAKED_SKIP_FACTOR = 10.0


def shift_parameter_names(
    shift_lever_id: int,
) -> tuple[str, str]:
    """Names of the two optimizable parameters holding the
    displacement of the shifted pair pivot
    (``shift = (dx, dy)``, millimetres)."""
    return (
        f"lever.{shift_lever_id}.shift.x",
        f"lever.{shift_lever_id}.shift.y",
    )


def _mean_squared_difference(
    target: TargetCurve,
    reference: TargetCurve,
    support_angles: tuple[float, ...],
) -> float:
    if not support_angles:
        return 0.0
    total = sum(
        (target.evaluate(angle) - reference.evaluate(angle))
        ** 2
        for angle in support_angles
    )
    return total / len(support_angles)


class PivotShiftOptimizer:
    """Evaluates candidates at the two endpoints of the pivot
    displacement (unbraked and braked).

    Parameters
    ----------
    builder:
        The (parameterless) PivotShiftBuilder used to build
        one mechanism per shift position.  A builder is
        created per candidate definition internally, so a
        single shared builder instance may be passed.
    simulator:
        Simulates the curve for each built mechanism.
    fitness:
        BrakeCurveFitness scoring all (result, target) pairs.
    definition:
        Base mechanism definition; candidate parameters are
        applied as an overlay (length, angle, pivot).
    targets:
        Exactly two target curves: unbraked (shift = 0) and
        braked.
    shift_lever_id:
        Id of one lever of the shifted pair.
    anchor_output:
        When True, the simulated output angles are shifted
        into the raw target-curve frame before scoring (same
        anchoring as the mixed brake optimizer).
    reference_output_angle:
        Raw target output of the unbraked curve at the middle
        support position (radians).
    shift_effect_weight:
        Fitness pressure keeping the pivot MOVE: a collapsed
        shift (braked == unbraked although the targets
        differ) pays the full quadratic deviation.
    """

    def __init__(
        self,
        *,
        builder: PivotShiftBuilder,
        simulator: MechanismSimulator,
        fitness: BrakeCurveFitness,
        definition: MechanismDefinition,
        targets: tuple[TargetCurve, ...],
        shift_lever_id: int,
        transmission_angle_constraint: (
            TransmissionAngleConstraint | None
        ) = None,
        anchor_output: bool = False,
        reference_output_angle: float = 0.0,
        shift_effect_weight: float = 1.0,
    ) -> None:
        if len(targets) != 2:
            raise ValueError(
                "Exactly two targets (unbraked, braked) "
                "are required for the pivot-shift "
                "optimization."
            )
        self._builder = builder
        self._simulator = simulator
        self._fitness = fitness
        self._definition = definition
        self._targets = tuple(targets)
        self._shift_lever_id = shift_lever_id
        self._shift_names = shift_parameter_names(
            shift_lever_id,
        )
        # The lever pair shares ONE common pivot: the
        # partner's pivot parameters are not independent
        # degrees of freedom, they always follow the leader
        # lever's pivot (also during optimization), so the
        # pair can never drift apart.
        self._pair_ids = resolve_pair_ids(
            definition,
            shift_lever_id,
        )
        self._transmission_constraint = (
            transmission_angle_constraint
        )
        self._shift_effect_weight = (
            shift_effect_weight
        )
        self._anchor_output = anchor_output
        self._reference_output_angle = (
            reference_output_angle
        )
        support_angles = tuple(
            self._simulator.motion
        )
        self._braked_skip_penalty = (
            BRAKED_SKIP_FACTOR
            * _mean_squared_difference(
                self._targets[1],
                self._targets[0],
                support_angles,
            )
        )
        self._cache: dict[
            tuple[ParameterSet, tuple[float, float]],
            SimulationResult,
        ] = {}
        self._mechanism_cache: dict[
            tuple[ParameterSet, tuple[float, float]],
            Mechanism,
        ] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._evaluations = 0
        self._build_errors = 0

    # ------------------------------------------------------------------
    # Pipeline (1)-(5)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: ParameterSet,
    ) -> float:
        """Evaluate one candidate at both shift endpoints and
        return the fitness (lower is better)."""
        self._evaluations += 1
        offset = (
            self._output_offset(parameters)
            if self._anchor_output
            else 0.0
        )
        shift = self.shift_for(parameters)
        pairs = []
        mechanisms = []
        unbraked = self._cached_simulation(
            parameters,
            (0.0, 0.0),
        )
        if unbraked is None:
            logger.debug(
                "Build error at the unbraked position for "
                "candidate %s; hard penalty.",
                parameters,
            )
            return PENALTY_MAX_BLOCKING
        pairs.append(
            (
                self._anchored_result(
                    unbraked,
                    offset,
                ),
                self._targets[0],
            )
        )
        skipped_penalty = 0.0
        braked = self._cached_simulation(
            parameters,
            shift,
        )
        if braked is None:
            skipped_penalty = (
                self._braked_skip_penalty
            )
            logger.debug(
                "Build error for candidate %s at shift=%s; "
                "braked position penalized.",
                parameters,
                shift,
            )
        else:
            pairs.append(
                (
                    self._anchored_result(
                        braked,
                        offset,
                    ),
                    self._targets[1],
                )
            )
        configured_weights = self._fitness.weights
        weights = (
            None
            if configured_weights is None
            else tuple(
                configured_weights[index]
                for index in range(len(pairs))
            )
        )
        fitness = self._fitness.evaluate(
            pairs,
            weights=weights,
        )
        fitness += skipped_penalty
        if self._shift_effect_weight != 0.0 and len(
            pairs,
        ) == 2:
            fitness += (
                self._shift_effect_weight
                * self._pair_shift_effect_deviation(
                    pairs,
                )
            )
        if self._transmission_constraint is not None:
            for position in ((0.0, 0.0), shift):
                mechanism = self._cached_mechanism(
                    parameters,
                    position,
                )
                if mechanism is not None:
                    mechanisms.append(mechanism)
            fitness += (
                self._transmission_constraint.evaluate(
                    mechanisms
                )
            )
        return fitness

    def shift_for(
        self,
        parameters: ParameterSet,
    ) -> tuple[float, float]:
        """The candidate's braked displacement of the pair
        pivot (``(0, 0)`` when the candidate carries no shift
        parameters)."""
        values = parameters.values()
        return (
            values.get(self._shift_names[0], 0.0),
            values.get(self._shift_names[1], 0.0),
        )

    # ------------------------------------------------------------------
    # Simulation plumbing (analogous to MixedBrakeOptimizer)
    # ------------------------------------------------------------------

    def _cached_mechanism(
        self,
        parameters: ParameterSet,
        shift: tuple[float, float],
    ) -> Mechanism | None:
        key = (parameters, shift)
        cached = self._mechanism_cache.get(key)
        if cached is not None:
            return cached
        try:
            builder = self._builder_for(parameters)
            mechanism = builder.build(shift)
        except MechanismBuildError:
            return None
        self._mechanism_cache[key] = mechanism
        return mechanism

    def _cached_simulation(
        self,
        parameters: ParameterSet,
        shift: tuple[float, float],
    ) -> SimulationResult | None:
        key = (parameters, shift)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        mechanism = self._cached_mechanism(
            parameters,
            shift,
        )
        if mechanism is None:
            self._build_errors += 1
            return None
        results = self._simulator.simulate(mechanism)
        if len(results) == 0:
            return None
        result = self._combined_result(results)
        self._cache[key] = result
        return result

    @staticmethod
    def _combined_result(
        results: tuple[SimulationResult, ...],
    ) -> SimulationResult:
        """Combine the per-stage simulations into one overall
        result (same convention as the mixed brake
        optimizer)."""
        if len(results) == 1:
            return results[0]
        blocked = next(
            (
                result.blocked_at
                for result in results
                if (
                    not result.success
                    and result.blocked_at is not None
                )
            ),
            None,
        )
        count = min(
            len(results[0].input_angles),
            len(results[-1].output_angles),
        )
        success = all(
            result.success for result in results
        )
        return SimulationResult(
            input_angles=(
                results[0].input_angles[:count]
            ),
            output_angles=(
                results[-1].output_angles[:count]
            ),
            success=success,
            blocked_at=(
                blocked if not success else None
            ),
        )

    def _builder_for(
        self,
        parameters: ParameterSet,
    ) -> PivotShiftBuilder:
        """Create a builder for the candidate's parameter
        overlay; falls back to the shared builder when the
        candidate matches the base definition."""
        definition = self.apply_parameters(
            parameters,
        )
        if definition == self._definition:
            return self._builder
        return PivotShiftBuilder(
            definition,
            shift_lever_id=self._shift_lever_id,
        )

    def apply_parameters(
        self,
        parameters: ParameterSet,
    ) -> MechanismDefinition:
        """Apply the candidate's geometry parameters (length,
        angle, pivot x/y/z) onto the base definition.  The
        pivot parameters of the pair partners follow the
        leader lever (common pivot); the shift parameters are
        NOT part of the definition: they only move the pair
        pivot of the braked build, so braked and unbraked
        share the identical definition apart from that
        displacement."""
        from dataclasses import replace

        from core.point3d import Point3D

        values = parameters.values()
        resolved: dict[int, Point3D] = {}
        leader_pivot = None
        for lever in self._definition.levers:
            pivot_x = values.get(
                f"lever.{lever.id}.pivot.x",
                lever.pivot.x,
            )
            pivot_y = values.get(
                f"lever.{lever.id}.pivot.y",
                lever.pivot.y,
            )
            pivot_z = values.get(
                f"lever.{lever.id}.pivot.z",
                lever.pivot.z,
            )
            resolved[lever.id] = Point3D(
                x=pivot_x,
                y=pivot_y,
                z=pivot_z,
            )
            if lever.id == self._shift_lever_id:
                leader_pivot = resolved[lever.id]
        if leader_pivot is None:
            raise KeyError(
                f"unknown shift lever {self._shift_lever_id}"
            )
        for lever_id in self._pair_ids:
            resolved[lever_id] = leader_pivot
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
            levers.append(
                replace(
                    lever,
                    length_start=length,
                    angle_start=angle,
                    pivot=resolved[lever.id],
                )
            )
        return MechanismDefinition(tuple(levers))

    def _pair_shift_effect_deviation(
        self,
        pairs,
    ) -> float:
        """Shift-effect deviation between the evaluated
        endpoints (first = unbraked, last = fully braked):
        how far the actual output change produced by the
        pivot displacement deviates from the change the
        target curves prescribe.  Zero for a pivot that
        really moves; maximal when the shift collapses."""
        first_result, first_target = pairs[0]
        last_result, last_target = pairs[-1]
        angles = first_result.input_angles
        count = min(
            len(angles),
            len(first_result.output_angles),
            len(last_result.output_angles),
        )
        if count == 0:
            return 0.0
        angles = angles[:count]
        target_diff = tuple(
            braked - unbraked
            for unbraked, braked in zip(
                first_target.sample(
                    angles,
                ).output_angles,
                last_target.sample(
                    angles,
                ).output_angles,
            )
        )
        actual_diff = tuple(
            braked - unbraked
            for unbraked, braked in zip(
                first_result.output_angles[:count],
                last_result.output_angles[:count],
            )
        )
        total = sum(
            (actual - target) ** 2
            for actual, target in zip(
                actual_diff,
                target_diff,
            )
        )
        return total / count

    def _output_offset(
        self,
        parameters: ParameterSet,
    ) -> float:
        lever = self._definition.get_lever(
            self._output_lever_id
        )
        angle = parameters.values().get(
            f"lever.{lever.id}.angle",
            lever.angle_start,
        )
        angle = self._clamp(
            angle,
            lever.angle_min,
            lever.angle_max,
        )
        return angle - self._reference_output_angle

    @staticmethod
    def _anchored_result(
        result: SimulationResult,
        offset: float,
    ) -> SimulationResult:
        if offset == 0.0:
            return result
        return SimulationResult(
            input_angles=result.input_angles,
            output_angles=tuple(
                angle - offset
                for angle in result.output_angles
            ),
            success=result.success,
            blocked_at=result.blocked_at,
        )

    @property
    def _output_lever_id(self) -> int:
        output_id = None
        for lever in self._definition.levers:
            if lever.driver is not None:
                output_id = lever.id
        if output_id is None:
            output_id = (
                self._definition.levers[-1].id
            )
        return output_id

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
