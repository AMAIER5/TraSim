"""
optimization/mechanism_optimizer.py

Adapter between mechanism simulation
and evolutionary optimization.

Cache key now uses immutable values instead of object references
to prevent incorrect cache hits between different mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization.mechanism_builder import MechanismBuilder
from optimization.fitness_function import FitnessFunction
from optimization.parameter_set import ParameterSet
from simulation.mechanism_simulator import MechanismSimulator


@dataclass(frozen=True, slots=True)
class OptimizationCacheKey:
    """
    Cache identity for one fitness evaluation.

    Uses IMMUTABLE VALUES only to ensure proper hashing:
    - parameters: the ParameterSet (already hashable)
    - motion_start, motion_max, motion_step: float values from MotionRange
    - precision: tuple of values if present, else None
    - stage_limit: int or None

    This prevents cache collisions between different mechanisms
    that might have the same object reference but different values.
    """

    parameters: ParameterSet

    # Motion range values (not the object reference)
    motion_start: float
    motion_max: float
    motion_step: float
    motion_direction: int

    precision: tuple[Any, ...] | None  # Convert precision to tuple if present

    stage_limit: int | None


class MechanismOptimizer:
    """
    Evaluates mechanism candidates.

    Fitness results are cached for identical optimization
    conditions. The cache key now uses immutable values
    to prevent incorrect cache hits.
    """


    def __init__(
        self,
        *,
        builder: MechanismBuilder,
        simulator: MechanismSimulator,
        fitness: FitnessFunction,
    ) -> None:

        self._builder = builder
        self._simulator = simulator
        self._fitness = fitness


        self._cache: dict[
            OptimizationCacheKey,
            float,
        ] = {}


        self._cache_hits = 0
        self._cache_misses = 0
        self._evaluations = 0



    def evaluate(
        self,
        parameters: ParameterSet,
    ) -> float:
        """
        Evaluate a mechanism candidate.

        Results are cached by parameters and
        simulation configuration.
        """

        self._evaluations += 1

        # Extract motion values as immutable floats (not object references)
        motion = self._simulator.motion
        motion_start = motion.start_angle
        motion_max = motion.max_angle
        motion_step = motion.step
        motion_direction = motion.direction

        # Extract precision as immutable tuple if present
        precision = self._simulator.precision
        if precision is not None:
            # Convert precision to tuple of its values for proper hashing
            precision_tuple = tuple(vars(precision).values())
        else:
            precision_tuple = None

        key = OptimizationCacheKey(
            parameters=parameters,
            motion_start=motion_start,
            motion_max=motion_max,
            motion_step=motion_step,
            motion_direction=motion_direction,
            precision=precision_tuple,
            stage_limit=self._simulator.stage_limit,
        )


        cached = self._cache.get(
            key
        )


        if cached is not None:

            self._cache_hits += 1

            return cached



        self._cache_misses += 1


        mechanism = self._builder.build(
            parameters
        )


        simulation = self._simulator.simulate(
            mechanism
        )


        result = self._fitness.evaluate(
            simulation
        )


        self._cache[
            key
        ] = result


        return result



    def clear_cache(
        self,
    ) -> None:
        """
        Remove all cached evaluations.
        """

        self._cache.clear()

        self._cache_hits = 0
        self._cache_misses = 0
        self._evaluations = 0



    def get_cache_stats(
        self,
    ) -> dict[str, int]:
        """
        Return cache statistics.
        """

        return {
            "evaluations": self._evaluations,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
