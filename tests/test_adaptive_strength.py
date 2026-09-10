"""
tests/test_adaptive_strength.py

Tests for the 1/5 success rule step-size controller.
"""

from __future__ import annotations

import random

from optimization.adaptive_strength import (
    AdaptiveStrength,
)
from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.parameter import (
    Parameter,
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
from optimization.reproduction import (
    Reproduction,
)


def make_mutation(strength: float = 0.1) -> ParameterMutation:

    return ParameterMutation(
        strength=strength,
        random_generator=random.Random(0),
    )


def test_invalid_window_rejected():
    """
    A non-positive window must raise ValueError.
    """

    try:

        AdaptiveStrength(
            make_mutation(),
            window=0,
        )

        assert False

    except ValueError:

        assert True


def test_invalid_target_rate_rejected():
    """
    The target success rate must lie strictly in (0, 1).
    """

    for rate in (0.0, 1.0, 1.5, -0.1):

        try:

            AdaptiveStrength(
                make_mutation(),
                target_success_rate=rate,
            )

            assert False

        except ValueError:

            assert True


def test_invalid_update_factor_rejected():
    """
    The update factor must be greater than 1 so that it
    can both increase and decrease the strength.
    """

    for factor in (0.5, 1.0, -1.0):

        try:

            AdaptiveStrength(
                make_mutation(),
                update_factor=factor,
            )

            assert False

        except ValueError:

            assert True


def test_invalid_bounds_rejected():
    """
    max_strength must be >= min_strength.
    """

    try:

        AdaptiveStrength(
            make_mutation(),
            min_strength=0.5,
            max_strength=0.1,
        )

        assert False

    except ValueError:

        assert True


def test_strength_increases_on_high_success_rate():
    """
    When the success rate exceeds the 1/5 target, the
    mutation strength must increase.
    """

    mutation = make_mutation(strength=0.1)

    controller = AdaptiveStrength(
        mutation,
        window=5,
        update_factor=1.5,
    )

    # 5 successes in a row -> rate 1.0 > 0.2 -> increase.
    for _ in range(5):

        controller.record(True)

    assert controller.strength > 0.1
    assert controller.history[-1] == controller.strength


def test_strength_decreases_on_low_success_rate():
    """
    When the success rate is below the 1/5 target, the
    mutation strength must decrease.
    """

    mutation = make_mutation(strength=0.1)

    controller = AdaptiveStrength(
        mutation,
        window=5,
        update_factor=1.5,
    )

    # 5 failures in a row -> rate 0.0 < 0.2 -> decrease.
    for _ in range(5):

        controller.record(False)

    assert controller.strength < 0.1


def test_strength_unchanged_at_target_rate():
    """
    When the success rate exactly equals the target, the
    strength must stay unchanged.
    """

    mutation = make_mutation(strength=0.1)

    controller = AdaptiveStrength(
        mutation,
        window=5,
        target_success_rate=0.2,
        update_factor=1.5,
    )

    # 1 success and 4 failures -> rate 0.2 == target.
    controller.record(True)

    for _ in range(4):

        controller.record(False)

    assert controller.strength == 0.1


def test_strength_clamped_to_min():
    """
    The strength must never drop below min_strength.
    """

    mutation = make_mutation(strength=0.1)

    controller = AdaptiveStrength(
        mutation,
        window=2,
        update_factor=2.0,
        min_strength=0.05,
    )

    # Many failures -> strength tries to collapse.
    for _ in range(20):

        controller.record(False)

    assert controller.strength >= 0.05


def test_strength_clamped_to_max():
    """
    The strength must never exceed max_strength.
    """

    mutation = make_mutation(strength=0.1)

    controller = AdaptiveStrength(
        mutation,
        window=2,
        update_factor=2.0,
        max_strength=0.3,
    )

    # Many successes -> strength tries to explode.
    for _ in range(20):

        controller.record(True)

    assert controller.strength <= 0.3


def test_adaptation_waits_for_full_window():
    """
    The strength must not change before the window is
    full.
    """

    mutation = make_mutation(strength=0.1)

    controller = AdaptiveStrength(
        mutation,
        window=5,
    )

    # Fewer than window outcomes: no adaptation yet.
    for _ in range(4):

        controller.record(True)

    assert controller.strength == 0.1


def test_engine_records_adaptation():
    """
    When an AdaptiveStrength controller is attached to
    the engine, the engine must feed it the per-
    generation improvement outcome so the mutation
    strength evolves over the run.
    """

    def create_candidate(value: float):

        return ParameterSet(
            (
                Parameter(
                    name="length",
                    minimum=0.0,
                    maximum=100.0,
                    value=value,
                ),
            )
        )

    # Fitness = the candidate's value, so improvement
    # happens whenever a smaller value is found.
    mutation = ParameterMutation(
        strength=0.2,
        random_generator=random.Random(1),
    )

    controller = AdaptiveStrength(
        mutation,
        window=5,
    )

    initial_strength = controller.strength

    engine = EvolutionEngine(
        population=Population(
            (
                create_candidate(90.0),
                create_candidate(80.0),
                create_candidate(70.0),
            )
        ),
        evaluator=lambda candidate: candidate.get("length").value,
        selection_count=1,
        reproduction=Reproduction(
            mutation=mutation,
            random_generator=random.Random(1),
        ),
        max_generations=20,
        adaptive_strength=controller,
    )

    for _ in engine.run(children_count=20):

        pass

    # The engine must have driven the controller: the
    # strength history must have grown beyond the initial
    # value, proving the controller was fed per-generation
    # outcomes and adapted the mutation strength.
    assert len(controller.history) > 1

    assert any(
        strength != initial_strength
        for strength in controller.history[1:]
    )


def test_engine_without_controller_is_unchanged():
    """
    Without an adaptive controller the engine must behave
    exactly as before (the mutation strength stays at its
    configured value).
    """

    def create_candidate(value: float):

        return ParameterSet(
            (
                Parameter(
                    name="length",
                    minimum=0.0,
                    maximum=100.0,
                    value=value,
                ),
            )
        )

    mutation = ParameterMutation(
        strength=0.15,
        random_generator=random.Random(1),
    )

    engine = EvolutionEngine(
        population=Population(
            (
                create_candidate(90.0),
                create_candidate(80.0),
            )
        ),
        evaluator=lambda candidate: candidate.get("length").value,
        selection_count=1,
        reproduction=Reproduction(
            mutation=mutation,
            random_generator=random.Random(1),
        ),
        max_generations=10,
    )

    for _ in engine.run(children_count=10):

        pass

    assert mutation.strength == 0.15
