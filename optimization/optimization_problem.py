"""
optimization/optimization_problem.py

Public entry point for mechanism optimization.
"""

from __future__ import annotations

import random
from random import Random

from analysis.curve_fitness import (
    CurveFitness,
)
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
from optimization.parameter_mutation import (
    ParameterMutation,
)
from optimization.parameter_set import (
    ParameterSet,
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
from simulation.mechanism_simulator import (
    MechanismSimulator,
)


class OptimizationProblem:
    """
    High-level optimization interface.

    This class wires together the complete optimization
    infrastructure behind one simple API.
    """

    def __init__(
        self,
        *,
        parameter_template: ParameterSet,
        simulator: MechanismSimulator,
        fitness: CurveFitness,
        builder: StandardMechanismBuilder | None = None,
        random_generator: Random | None = None,
    ) -> None:

        self._parameter_template = parameter_template

        self._simulator = simulator

        self._fitness = fitness

        self._builder = (
            builder
            if builder is not None
            else StandardMechanismBuilder()
        )

        self._random = (
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
            random_generator=self._random,
        ).create(
            self._parameter_template,
            size=population_size,
        )

        optimizer = MechanismOptimizer(
            builder=self._builder,
            simulator=self._simulator,
            fitness=self._fitness,
        )

        reproduction = Reproduction(
            mutation=ParameterMutation(
                random_generator=self._random,
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