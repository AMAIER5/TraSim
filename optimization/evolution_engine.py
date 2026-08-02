"""
optimization/evolution_engine.py

High level evolutionary optimization loop.

Supports:
- evaluation
- selection
- reproduction
- fitness based early stopping
- stagnation detection
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.population import (
    Population,
)

from optimization.reproduction import (
    Reproduction,
)

from optimization.selection import (
    Selection,
)


class EvolutionEngine:
    """
    Executes evolutionary optimization.

    The engine coordinates:

    - evaluation
    - selection
    - reproduction

    Additionally it can stop automatically when:

    - a target fitness is reached
    - no relevant improvement occurs
    """

    def __init__(
        self,
        *,
        population: Population,
        evaluator: Callable[
            [ParameterSet],
            float,
        ],
        selection_count: int,
        reproduction: Reproduction,
        target_fitness: float | None = None,
        max_generations: int = 100,
        stagnation_limit: int | None = None,
        stagnation_tolerance: float = 1e-6,
    ) -> None:

        self.population = population

        self.evaluator = evaluator

        self.selection_count = (
            selection_count
        )

        self.selection = Selection()

        self.reproduction = reproduction

        # stopping criteria

        self.target_fitness = (
            target_fitness
        )

        self.max_generations = (
            max_generations
        )

        self.stagnation_limit = (
            stagnation_limit
        )

        self.stagnation_tolerance = (
            stagnation_tolerance
        )
   
        # state

        self.scores: dict[
            ParameterSet,
            float,
        ] = {}

        self.best_candidate: ParameterSet | None = None

        self.best_score = float("inf")

        self._stagnation_counter = 0

        self.stop_reason: str | None = None


    def evaluate_population(
        self,
    ) -> None:
        """
        Evaluate current population.
        """

        self.scores = {
            candidate:
                self.evaluator(candidate)
            for candidate
            in self.population
        }


    def update_best(
        self,
    ) -> None:
        """
        Update best known solution.

        Returns True if improvement occurred.
        """

        candidate, score = min(
            self.scores.items(),
            key=lambda item: item[1],
        )

        improvement = (
            self.best_score
            -
            score
        )

        if improvement >= self.stagnation_tolerance:

            self.best_score = score
            self.best_candidate = candidate

            self._stagnation_counter = 0

        else:

            self._stagnation_counter += 1
            

    def should_stop(self) -> bool:
        """
        Check stopping criteria and store reason.
        """

        if (
            self.target_fitness is not None
            and
            self.best_score <= self.target_fitness
        ):
            self.stop_reason = (
                "target_fitness_reached"
            )
            return True


        if (
            self.stagnation_limit is not None
            and
            self._stagnation_counter
            >= self.stagnation_limit
        ):
            self.stop_reason = (
                "stagnation_limit_reached"
            )
            return True


        return False


    def step(
        self,
        *,
        children_count: int,
    ) -> Population:
        """
        Execute one evolutionary generation.
        """

        self.evaluate_population()

        survivors = self.selection.select(
            self.population,
            self.scores,
            count=self.selection_count,
        )

        next_population = (
            self.reproduction.create(
                survivors,
                count=children_count,
            )
        )

        self.population = (
            next_population
        )

        return next_population


    def run(
        self,
        *,
        children_count: int,
    ) -> Iterator[int]:
        """
        Run evolutionary optimization.
        """

        for generation in range(
            self.max_generations
        ):

            if not self.scores:
                self.evaluate_population()

            self.update_best()

            yield generation

            if self.should_stop():
                break

            self.step(
                children_count=children_count,
            )
            
        if self.stop_reason is None:

            self.stop_reason = (
                "max_generations_reached"
            )