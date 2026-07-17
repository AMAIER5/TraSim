"""
tests/test_parameter_mutation.py

Tests for parameter mutation.
"""

from __future__ import annotations

import random

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_set import (
    ParameterSet,
)

from optimization.parameter_mutation import (
    ParameterMutation,
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