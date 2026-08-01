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

    def evaluate(
        self,
        parameters: ParameterSet,
    ) -> float:

        mechanism = self._builder.build(
            parameters
        )

        simulation = self._simulator.simulate(
            mechanism
        )

        return self._fitness.evaluate(
            simulation
        )