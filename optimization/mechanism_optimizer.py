"""
optimization/mechanism_optimizer.py

Adapter between mechanism simulation
and evolutionary optimization.
"""

from __future__ import annotations

from optimization.mechanism_builder import MechanismBuilder
from optimization.fitness_function import (
    FitnessFunction,
)
from optimization.parameter_set import (
    ParameterSet,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)


class MechanismOptimizer:
    """
    Evaluates mechanism candidates.

    Fitness results are cached for identical parameter sets.
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
            ParameterSet,
            float,
        ] = {}

        self._cache_hits = 0
        self._cache_misses = 0


    def evaluate(
        self,
        parameters: ParameterSet,
    ) -> float:
        """
        Evaluate a mechanism candidate.

        Results are cached by parameter set.
        """

        cached = self._cache.get(
            parameters
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
            parameters
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


    def get_cache_stats(
        self,
    ) -> dict[str, int]:
        """
        Return cache statistics.
        """

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }