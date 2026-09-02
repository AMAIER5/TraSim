"""
tests/test_reproduction.py

Tests for reproduction of candidates.
"""

from __future__ import annotations

import random

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


def create_candidate(
    value: float,
):

    return ParameterSet(
        (
            Parameter(
                name="length",
                minimum=10.0,
                maximum=100.0,
                value=value,
            ),
        )
    )


def test_reproduction_creates_children():

    population = Population(
        (
            create_candidate(50.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(1)
        ),
        random_generator=random.Random(1),
    )

    children = reproduction.create(
        population,
        count=3,
    )

    assert len(children) == 3


def test_reproduction_keeps_parameters_valid():

    population = Population(
        (
            create_candidate(50.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(2)
        ),
        random_generator=random.Random(2),
    )

    children = reproduction.create(
        population,
        count=10,
    )

    for child in children:

        parameter = child.get(
            "length"
        )

        assert (
            parameter.minimum
            <=
            parameter.value
            <=
            parameter.maximum
        )


def test_reproduction_does_not_change_parent():

    parent = create_candidate(
        50.0
    )

    population = Population(
        (
            parent,
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(1)
        ),
        random_generator=random.Random(1),
    )

    reproduction.create(
        population,
        count=1,
    )

    assert parent.get(
        "length"
    ).value == 50.0


# ---------------------------------------------------------------------------
# Issue #10: random parent selection instead of deterministic round-robin
# ---------------------------------------------------------------------------


def test_reproduction_uses_random_parent_selection():
    """
    Issue #10: Reproduction must select parents randomly,
    not via deterministic round-robin.  With a single
    parent the result is the same, but with multiple
    survivors the distribution of parents must be
    non-deterministic (seed-dependent).
    """

    population = Population(
        (
            create_candidate(10.0),
            create_candidate(50.0),
            create_candidate(90.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            strength=0.0,
            random_generator=random.Random(0),
        ),
        random_generator=random.Random(42),
    )

    children = reproduction.create(
        population,
        count=20,
    )

    # With strength=0, each child's value equals its
    # parent's value.  Collect the unique parent values
    # to verify that more than one parent was chosen.
    values = {
        child.get("length").value
        for child in children
    }

    assert len(values) > 1


def test_reproduction_does_not_collapse_with_small_selection():
    """
    Issue #10: With deterministic round-robin, a small
    survivor count always produces the same parent
    sequence.  Random selection avoids this collapse.
    Here we verify that two independent runs with
    different seeds produce different distributions.
    """

    population = Population(
        (
            create_candidate(10.0),
            create_candidate(20.0),
        )
    )

    reproduction_a = Reproduction(
        mutation=ParameterMutation(
            strength=0.0,
            random_generator=random.Random(0),
        ),
        random_generator=random.Random(1),
    )

    reproduction_b = Reproduction(
        mutation=ParameterMutation(
            strength=0.0,
            random_generator=random.Random(0),
        ),
        random_generator=random.Random(999),
    )

    children_a = reproduction_a.create(
        population,
        count=30,
    )
    children_b = reproduction_b.create(
        population,
        count=30,
    )

    values_a = [
        child.get("length").value
        for child in children_a
    ]
    values_b = [
        child.get("length").value
        for child in children_b
    ]

    # Different seeds → different parent sequences.
    assert values_a != values_b


def test_reproduction_rejects_zero_count():
    """
    count must be positive.
    """

    population = Population(
        (
            create_candidate(50.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(0)
        ),
    )

    try:

        reproduction.create(
            population,
            count=0,
        )

        assert False

    except ValueError:

        assert True


def test_reproduction_rejects_negative_count():
    """
    Negative count must raise ValueError.
    """

    population = Population(
        (
            create_candidate(50.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            random_generator=random.Random(0)
        ),
    )

    try:

        reproduction.create(
            population,
            count=-5,
        )

        assert False

    except ValueError:

        assert True


def test_reproduction_with_multiple_parents_covers_all():
    """
    Issue #10: With enough children and multiple
    survivors, every survivor should be chosen at
    least once (probabilistic, but with count=100
    and 3 survivors this is virtually certain).
    """

    population = Population(
        (
            create_candidate(10.0),
            create_candidate(50.0),
            create_candidate(90.0),
        )
    )

    reproduction = Reproduction(
        mutation=ParameterMutation(
            strength=0.0,
            random_generator=random.Random(0),
        ),
        random_generator=random.Random(7),
    )

    children = reproduction.create(
        population,
        count=100,
    )

    values = {
        child.get("length").value
        for child in children
    }

    assert values == {10.0, 50.0, 90.0}