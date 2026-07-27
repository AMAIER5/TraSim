"""
optimization/mechanism_optimizer.py

Adapter between mechanism simulation
and evolutionary optimization.
"""

from __future__ import annotations

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)
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
        builder: StandardMechanismBuilder,
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