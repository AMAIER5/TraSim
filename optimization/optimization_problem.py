"""
optimization/optimization_problem.py

Public entry point for mechanism optimization.
"""

from __future__ import annotations

import random
from random import Random
from typing import Any, Callable

from mechanics.mechanism_factory import MechanismFactory
from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)

from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)
from optimization.optimizer_runner import (
    OptimizerRunner,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.parameter_mutation import (
    ParameterMutation,
)
from optimization.population import (
    Population,
)
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import (
    Reproduction,
)


class OptimizationProblem:
    """
    High-level optimization interface.

    This class wires together all optimization
    infrastructure behind one simple API.
    """

    def __init__(
        self,
        *,
        parameter_template: ParameterSet,
        simulator: Callable[[Any], Any],
        fitness: Callable[[Any], float],
        builder: Any | None = None,
        random_generator: Random | None = None,
    ):

        self.parameter_template = (
            parameter_template
        )

        self.simulator = simulator

        self.fitness = fitness

        self.builder = (
            builder
            if builder is not None
            else StandardMechanismBuilder()
        )

        self.random = (
            random_generator
            if random_generator is not None
            else random.Random()
        )

    def optimize(
        self,
        *,
        population_size: int,
        generations: int,
        children_per_generation: int,
        selection_count: int = 5,
    ) -> Population:
        """
        Execute a complete optimization run.
        """

        population = PopulationFactory(
            random_generator=self.random,
        ).create(
            self.parameter_template,
            size=population_size,
        )

        mechanism_factory = (
            MechanismFactory(
                builder=self.builder.build,
            )
        )

        optimizer = MechanismOptimizer(
            mechanism_factory=(
                mechanism_factory.create
            ),
            simulator=self.simulator,
            fitness=self.fitness,
        )

        reproduction = Reproduction(
            mutation=ParameterMutation(
                random_generator=self.random,
            )
        )

        engine = EvolutionEngine(
            population=population,
            evaluator=optimizer.evaluate,
            selection_count=selection_count,
            reproduction=reproduction,
        )

        runner = OptimizerRunner(
            engine=engine,
        )

        return runner.run(
            generations=generations,
            children_count=children_per_generation,
        )