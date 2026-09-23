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

import os
from collections.abc import Callable, Iterator
from concurrent.futures import (
    ProcessPoolExecutor,
)

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


_WORKER_EVALUATOR: Callable[
    [ParameterSet],
    float,
] | None = None

_WORKER_STATS_PROVIDER: (
    Callable[[], dict[str, int]] | None
) = None


def _worker_initializer(
    evaluator_factory: Callable[
        [],
        Callable[[ParameterSet], float],
    ],
    stats_provider: Callable[
        [],
        dict[str, int]
    ] | None,
) -> None:
    """
    Initialize one worker process.

    Builds the evaluator (builder/simulator/
    fitness) ONCE per process so that every
    evaluation reuses the already initialized
    infrastructure instead of rebuilding it.
    """

    global _WORKER_EVALUATOR, _WORKER_STATS_PROVIDER

    _WORKER_EVALUATOR = (
        evaluator_factory()
    )

    _WORKER_STATS_PROVIDER = (
        stats_provider
    )


def _evaluate_candidate(
    candidate: ParameterSet,
) -> float:
    """
    Evaluate one candidate inside a worker
    process using the process-local evaluator.
    """

    if _WORKER_EVALUATOR is None:
        raise RuntimeError(
            "worker evaluator not initialized"
        )

    return _WORKER_EVALUATOR(
        candidate
    )


def _worker_stats(
    process_index: int,
) -> tuple[int, dict[str, int]]:
    """
    Collect the cumulative stats of one worker
    process, tagged with the process id so the
    main process can de-duplicate answers.
    """

    if _WORKER_STATS_PROVIDER is None:
        return (
            os.getpid(),
            {},
        )

    return (
        os.getpid(),
        _WORKER_STATS_PROVIDER(),
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

    Parallel evaluation
    -------------------

    With ``workers=1`` (default) the population is
    evaluated sequentially and ``evaluator`` is
    used directly.  With ``workers > 1`` a process
    pool evaluates ``evaluate_population()`` via
    an order-preserving ``pool.map`` so the score
    mapping stays DETERMINISTIC and identical to
    the sequential mode.

    The worker processes build their own evaluator
    through ``evaluator_factory`` (builder/simulator/
    fitness) ONCE per process via the pool
    initializer.  The main-process ``evaluator`` is
    not used in parallel mode.

    Statistics collected inside worker processes
    (e.g. solver statistics) are NOT visible in the
    main process automatically.  An optional
    ``stats_provider`` can expose them; its
    per-worker cumulative values can be retrieved
    aggregated with ``collect_worker_stats()``.
    """

    def __init__(
        self,
        *,
        population: Population,
        evaluator: Callable[
            [ParameterSet],
            float,
        ] | None = None,
        selection_count: int,
        reproduction: Reproduction,
        target_fitness: float | None = None,
        max_generations: int = 100,
        stagnation_limit: int | None = None,
        stagnation_tolerance: float = 1e-6,
        adaptive_strength=None,
        workers: int = 1,
        evaluator_factory: Callable[
            [],
            Callable[[ParameterSet], float],
        ] | None = None,
        stats_provider: Callable[
            [],
            dict[str, int]
        ] | None = None,
    ) -> None:

        if workers < 1:
            raise ValueError(
                "workers must be at least 1"
            )

        if (
            workers > 1
            and
            evaluator_factory is None
        ):
            raise ValueError(
                "workers > 1 requires an "
                "evaluator_factory"
            )

        if (
            workers == 1
            and
            evaluator is None
        ):
            raise ValueError(
                "sequential mode (workers == 1) "
                "requires an evaluator"
            )

        self.workers = workers

        self.evaluator_factory = (
            evaluator_factory
        )

        self.stats_provider = (
            stats_provider
        )

        self._pool = None

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

        # Optional adaptive step-size controller (1/5
        # success rule).  When provided, the engine feeds
        # it the per-generation improvement outcome so it
        # can adapt the mutation strength automatically.
        self.adaptive_strength = (
            adaptive_strength
        )

        # state

        self.scores: dict[
            ParameterSet,
            float,
        ] = {}

        self.best_candidate: ParameterSet | None = None

        self.best_score = float("inf")

        self._stagnation_counter = 0

        self._improved_this_generation = False

        self.stop_reason: str | None = None

    def evaluate_population(
        self,
    ) -> None:
        """
        Evaluate current population.

        In parallel mode (``workers > 1``) the
        candidates are evaluated by a process pool
        using ``pool.map``, which PRESERVES the input
        order.  The resulting score mapping is
        therefore identical to the sequential mode
        and the optimization stays deterministic.
        """

        candidates = (
            self.population.members
        )

        if self.workers == 1:

            self.scores = {
                candidate:
                    self.evaluator(candidate)
                for candidate
                in candidates
            }

            return

        pool = self._ensure_pool()

        scores = pool.map(
            _evaluate_candidate,
            candidates,
        )

        self.scores = {
            candidate: score
            for candidate, score
            in zip(
                candidates,
                scores,
            )
        }

    def _ensure_pool(
        self,
    ) -> ProcessPoolExecutor:
        """
        Lazily create the worker pool so the worker
        processes (and their evaluator built by the
        initializer) are reused across generations
        instead of being recreated per evaluation.
        """

        if self._pool is None:

            self._pool = (
                ProcessPoolExecutor(
                    max_workers=self.workers,
                    initializer=(
                        _worker_initializer
                    ),
                    initargs=(
                        self.evaluator_factory,
                        self.stats_provider,
                    ),
                )
            )

        return self._pool

    def collect_worker_stats(
        self,
    ) -> dict[str, int]:
        """
        Aggregate cumulative worker statistics.

        Asks every worker process for the stats
        reported by ``stats_provider`` and sums the
        per-process values.  Answers are de-duplicated
        by process id so a worker answering more than
        once is not counted twice.

        Returns an empty mapping in sequential mode
        (``workers == 1``): in that mode all stats live
        in the main process anyway.
        """

        if self._pool is None:
            return {}

        results = self._pool.map(
            _worker_stats,
            range(
                4 * self.workers
            ),
            chunksize=1,
        )

        per_process: dict[int, dict[str, int]] = {}

        for process_id, stats in results:
            per_process[process_id] = stats

        totals: dict[str, int] = {}

        for stats in per_process.values():
            for key, value in stats.items():
                totals[key] = (
                    totals.get(key, 0)
                    +
                    value
                )

        return totals

    def close(
        self,
    ) -> None:
        """
        Shut down the worker pool.

        Sequential engines have no pool; the call is
        a no-op for them.
        """

        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def update_best(
        self,
    ) -> None:
        """
        Update best known solution.

        Issue #5: The previous docstring said "Returns True"
        but the method returned ``None``.  The docstring is
        now corrected.  The method still mutates
        ``self.best_candidate`` and ``self.best_score`` and
        manages the stagnation counter as a side effect.
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

            self._improved_this_generation = True

        else:

            self._stagnation_counter += 1

            self._improved_this_generation = False

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

        Issue #5: The previous loop evaluated and yielded,
        then stepped (creating children), but never
        re-evaluated the children produced by the final
        step.  This meant ``best_candidate`` could be one
        generation stale.

        The loop is restructured so that after every
        ``step()`` — including the last —
        ``evaluate_population()`` + ``update_best()`` run
        on the new population before ``should_stop()`` is
        checked.  This guarantees the best reported
        candidate reflects the final generation.
        """

        # Evaluate the initial population and set the
        # initial best before yielding generation 0.
        self.evaluate_population()
        self.update_best()

        for generation in range(
            self.max_generations
        ):
            yield generation

            if self.should_stop():
                break

            self.step(
                children_count=children_count,
            )

            # Issue #5: Re-evaluate the new population so
            # that the best candidate is never stale.
            self.evaluate_population()
            self.update_best()

            # Adapt the mutation strength based on whether
            # this generation improved the best solution.
            if self.adaptive_strength is not None:

                self.adaptive_strength.record(
                    self._improved_this_generation
                )

        if self.stop_reason is None:

            self.stop_reason = (
                "max_generations_reached"
            )