"""
tests/test_parameter_mutation.py

Tests for parameter mutation.
"""

from __future__ import annotations

import random

from optimization.parameter import (
    Parameter,
)
from optimization.parameter_mutation import (
    GAUSS,
    UNIFORM,
    ParameterMutation,
)
from optimization.parameter_set import (
    ParameterSet,
)


def create_parameter_set():

    return ParameterSet(
        (
            Parameter(
                name="rod_length",
                minimum=10.0,
                maximum=100.0,
                value=50.0,
            ),
            Parameter(
                name="lever_length",
                minimum=20.0,
                maximum=80.0,
                value=40.0,
            ),
        )
    )


def test_mutation_changes_parameter():

    mutation = ParameterMutation(
        random_generator=random.Random(1)
    )

    original = create_parameter_set()

    mutated = mutation.apply(
        original
    )

    assert mutated != original


def test_original_parameter_set_is_unchanged():

    original = create_parameter_set()

    mutation = ParameterMutation(
        random_generator=random.Random(1)
    )

    mutation.apply(
        original
    )

    assert original.get(
        "rod_length"
    ).value == 50.0


def test_mutated_values_stay_in_range():

    mutation = ParameterMutation(
        random_generator=random.Random(2)
    )

    result = mutation.apply(
        create_parameter_set()
    )

    for parameter in result.parameters:

        assert (
            parameter.minimum
            <=
            parameter.value
            <=
            parameter.maximum
        )


def test_zero_strength_keeps_values():

    mutation = ParameterMutation(
        strength=0.0,
        random_generator=random.Random(1),
    )

    original = create_parameter_set()

    result = mutation.apply(
        original
    )

    assert result == original


def test_default_distribution_is_gaussian():

    mutation = ParameterMutation(
        random_generator=random.Random(1)
    )

    assert mutation.distribution == GAUSS


def test_uniform_distribution_is_selectable():

    mutation = ParameterMutation(
        distribution=UNIFORM,
        random_generator=random.Random(1),
    )

    assert mutation.distribution == UNIFORM


def test_invalid_distribution_rejected():

    try:

        ParameterMutation(
            distribution="cauchy",
            random_generator=random.Random(1),
        )

        assert False

    except ValueError:

        assert True


def test_reflection_keeps_values_in_range():
    """
    Even with a large strength, every mutated value must
    stay inside the parameter range.  Reflection (not
    clamping) is used, so values near the boundary are
    mirrored back into the interior instead of piling up
    at the limits.
    """

    mutation = ParameterMutation(
        strength=1.0,
        random_generator=random.Random(7),
    )

    result = mutation.apply(
        create_parameter_set()
    )

    for parameter in result.parameters:

        assert parameter.minimum <= parameter.value <= parameter.maximum


def test_reflection_does_not_clamp_at_boundary():
    """
    A clamping mutation would place every value that
    overshoots exactly on the boundary.  Reflection
    instead mirrors it back into the interior, so over
    many large mutations we must see values strictly
    inside the range and not concentrated at the limits.
    """

    parameter = Parameter(
        name="x",
        minimum=0.0,
        maximum=100.0,
        value=99.0,
    )

    mutation = ParameterMutation(
        strength=0.5,
        random_generator=random.Random(123),
    )

    values = []

    for _ in range(500):

        mutated = mutation.apply(
            ParameterSet((parameter,))
        )

        values.append(
            mutated.get("x").value
        )

        parameter = mutated.get("x")

    on_boundary = sum(
        1
        for value in values
        if value == 0.0 or value == 100.0
    )

    # Reflection produces interior values far more often
    # than clamping, which would concentrate the mass at
    # 100.0 for a parent near the upper limit.
    assert on_boundary < len(values) / 2


def test_gaussian_mutation_is_centred():
    """
    A Gaussian mutation is centred on the parent value:
    the mean of many mutations must be close to the
    parent value (no systematic drift).
    """

    parameter = Parameter(
        name="x",
        minimum=0.0,
        maximum=100.0,
        value=50.0,
    )

    mutation = ParameterMutation(
        strength=0.2,
        random_generator=random.Random(2024),
    )

    samples = []

    for _ in range(2000):

        mutated = mutation.apply(
            ParameterSet((parameter,))
        )

        samples.append(
            mutated.get("x").value
        )

    mean = sum(samples) / len(samples)

    assert abs(mean - 50.0) < 5.0


def test_gaussian_distribution_uses_gauss_rng():
    """
    When distribution is GAUSS, the mutation must draw
    its delta from the Gaussian RNG path (many small
    steps, few large steps) instead of the uniform box
    path.  This is verified by spying on the random
    generator used.
    """

    class _SpyRng:

        def __init__(self):

            self.gauss_calls = 0
            self.uniform_calls = 0

        def gauss(self, mu, sigma):

            self.gauss_calls += 1

            return 0.0

        def uniform(self, a, b):

            self.uniform_calls += 1

            return 0.0

    spy = _SpyRng()

    mutation = ParameterMutation(
        strength=0.1,
        distribution=GAUSS,
        random_generator=spy,
    )

    mutation.apply(create_parameter_set())

    assert spy.gauss_calls == 2
    assert spy.uniform_calls == 0


def test_uniform_distribution_uses_uniform_rng():

    class _SpyRng:

        def __init__(self):

            self.gauss_calls = 0
            self.uniform_calls = 0

        def gauss(self, mu, sigma):

            self.gauss_calls += 1

            return 0.0

        def uniform(self, a, b):

            self.uniform_calls += 1

            return 0.0

    spy = _SpyRng()

    mutation = ParameterMutation(
        strength=0.1,
        distribution=UNIFORM,
        random_generator=spy,
    )

    mutation.apply(create_parameter_set())

    assert spy.uniform_calls == 2
    assert spy.gauss_calls == 0


def test_zero_strength_ignores_distribution():

    mutation = ParameterMutation(
        strength=0.0,
        distribution=UNIFORM,
        random_generator=random.Random(1),
    )

    original = create_parameter_set()

    assert mutation.apply(original) == original