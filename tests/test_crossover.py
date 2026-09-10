"""
tests/test_crossover.py

Tests for the crossover operator.
"""

from __future__ import annotations

import random

from optimization.crossover import (
    GROUPED,
    UNIFORM,
    Crossover,
    _lever_key,
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


def create_parent(value_a: float, value_b: float):
    """
    Create a two-parameter parent set.

    Parameters are named lever.1.length and lever.1.angle
    so that grouped crossover keeps them together.
    """

    return ParameterSet(
        (
            Parameter(
                name="lever.1.length",
                minimum=0.0,
                maximum=100.0,
                value=value_a,
            ),
            Parameter(
                name="lever.1.angle",
                minimum=0.0,
                maximum=100.0,
                value=value_b,
            ),
        )
    )


def create_two_lever_parent(
    lever1_length: float,
    lever1_angle: float,
    lever2_length: float,
    lever2_angle: float,
):
    """
    Create a four-parameter parent covering two levers.
    """

    return ParameterSet(
        (
            Parameter(
                name="lever.1.length",
                minimum=0.0,
                maximum=100.0,
                value=lever1_length,
            ),
            Parameter(
                name="lever.1.angle",
                minimum=0.0,
                maximum=100.0,
                value=lever1_angle,
            ),
            Parameter(
                name="lever.2.length",
                minimum=0.0,
                maximum=100.0,
                value=lever2_length,
            ),
            Parameter(
                name="lever.2.angle",
                minimum=0.0,
                maximum=100.0,
                value=lever2_angle,
            ),
        )
    )


def test_invalid_strategy_rejected():
    """
    An unknown crossover strategy must raise ValueError.
    """

    try:

        Crossover(
            strategy="arithmetic",
            random_generator=random.Random(0),
        )

        assert False

    except ValueError:

        assert True


def test_lever_key_groups_by_lever_id():
    """
    Parameters of one lever share the same group key; a
    parameter without the lever prefix is its own group.
    """

    assert _lever_key("lever.1.length") == "lever.1"
    assert _lever_key("lever.1.angle") == "lever.1"
    assert _lever_key("lever.2.length") == "lever.2"
    assert _lever_key("rod_length") == "rod_length"


def test_recombine_produces_valid_parameter_set():
    """
    A recombined child must have the same parameter names
    and stay within the parameter ranges.
    """

    parent_a = create_parent(10.0, 20.0)
    parent_b = create_parent(80.0, 90.0)

    crossover = Crossover(
        random_generator=random.Random(1),
    )

    child = crossover.recombine(parent_a, parent_b)

    names = [
        parameter.name
        for parameter in child.parameters
    ]

    assert names == ["lever.1.length", "lever.1.angle"]

    for parameter in child.parameters:

        assert parameter.minimum <= parameter.value <= parameter.maximum


def test_recombine_rejects_mismatched_parents():
    """
    Parents with different parameter names must raise
    ValueError.
    """

    parent_a = ParameterSet(
        (
            Parameter(
                name="a",
                minimum=0.0,
                maximum=100.0,
                value=50.0,
            ),
        )
    )

    parent_b = ParameterSet(
        (
            Parameter(
                name="b",
                minimum=0.0,
                maximum=100.0,
                value=50.0,
            ),
        )
    )

    crossover = Crossover(
        random_generator=random.Random(1),
    )

    try:

        crossover.recombine(parent_a, parent_b)

        assert False

    except ValueError:

        assert True


def test_uniform_crossover_uses_both_parents():
    """
    Uniform crossover draws each parameter from either
    parent.  With two clearly separated parents and many
    recombinations, both parents must contribute.
    """

    parent_a = create_parent(0.0, 0.0)
    parent_b = create_parent(100.0, 100.0)

    crossover = Crossover(
        strategy=UNIFORM,
        random_generator=random.Random(7),
    )

    sources = set()

    for _ in range(50):

        child = crossover.recombine(parent_a, parent_b)

        for parameter in child.parameters:

            if parameter.value == 0.0:

                sources.add("a")

            elif parameter.value == 100.0:

                sources.add("b")

    assert sources == {"a", "b"}


def test_grouped_crossover_keeps_lever_together():
    """
    Grouped crossover always inherits all parameters of one
    lever from the same parent.  With parents whose two
    parameters are (0, 10) and (90, 100), a grouped child is
    always either fully (0, 10) or fully (90, 100) - never a
    mix of the two within one lever.
    """

    parent_a = create_parent(0.0, 10.0)
    parent_b = create_parent(90.0, 100.0)

    crossover = Crossover(
        strategy=GROUPED,
        random_generator=random.Random(42),
    )

    for _ in range(100):

        child = crossover.recombine(parent_a, parent_b)

        values = tuple(
            parameter.value
            for parameter in child.parameters
        )

        assert values in {(0.0, 10.0), (90.0, 100.0)}


def test_grouped_crossover_mixes_between_levers():
    """
    Across two levers, grouped crossover must be able to
    take lever 1 from one parent and lever 2 from the
    other.  With enough recombinations all four
    combinations must appear.
    """

    parent_a = create_two_lever_parent(
        0.0, 0.0, 0.0, 0.0,
    )

    parent_b = create_two_lever_parent(
        100.0, 100.0, 100.0, 100.0,
    )

    crossover = Crossover(
        strategy=GROUPED,
        random_generator=random.Random(99),
    )

    seen = set()

    for _ in range(200):

        child = crossover.recombine(parent_a, parent_b)

        values = tuple(
            parameter.value
            for parameter in child.parameters
        )

        # Group signature: which parent each lever came from.
        lever1 = "a" if values[0] == 0.0 else "b"
        lever2 = "a" if values[2] == 0.0 else "b"

        seen.add((lever1, lever2))

    assert seen == {
        ("a", "a"),
        ("a", "b"),
        ("b", "a"),
        ("b", "b"),
    }


# ---------------------------------------------------------------------------
# Reproduction integration with crossover
# ---------------------------------------------------------------------------


def test_reproduction_without_crossover_is_mutations_only():
    """
    Without a crossover operator, reproduction must behave
    as before: each child is a mutated copy of a single
    parent.
    """

    parent = create_parent(50.0, 50.0)

    population = Population((parent,))

    reproduction = Reproduction(
        mutation=ParameterMutation(
            strength=0.0,
            random_generator=random.Random(0),
        ),
        random_generator=random.Random(0),
    )

    children = reproduction.create(
        population,
        count=3,
    )

    for child in children:

        assert child.get("lever.1.length").value == 50.0
        assert child.get("lever.1.angle").value == 50.0


def test_reproduction_with_crossover_combines_parents():
    """
    With crossover enabled, a child can combine parameters
    from two different parents.  Using strength=0 mutation
    (so the child equals the recombination) and two
    parents with disjoint values, some child must carry
    values from both parents.
    """

    parent_a = create_two_lever_parent(
        0.0, 0.0, 0.0, 0.0,
    )

    parent_b = create_two_lever_parent(
        100.0, 100.0, 100.0, 100.0,
    )

    population = Population(
        (
            parent_a,
            parent_b,
        )
    )

    mutation = ParameterMutation(
        strength=0.0,
        random_generator=random.Random(0),
    )

    reproduction = Reproduction(
        mutation=mutation,
        random_generator=random.Random(3),
        crossover=Crossover(
            strategy=UNIFORM,
            random_generator=random.Random(3),
        ),
    )

    mixed = False

    for _ in range(100):

        children = reproduction.create(
            population,
            count=10,
        )

        for child in children:

            values = tuple(
                parameter.value
                for parameter in child.parameters
            )

            if 0.0 in values and 100.0 in values:

                mixed = True

                break

        if mixed:

            break

    assert mixed


def test_reproduction_with_crossover_keeps_values_valid():
    """
    Crossover children stay within the parameter ranges
    after recombination and mutation.
    """

    parent_a = create_two_lever_parent(
        10.0, 20.0, 30.0, 40.0,
    )

    parent_b = create_two_lever_parent(
        80.0, 90.0, 70.0, 60.0,
    )

    population = Population(
        (
            parent_a,
            parent_b,
        )
    )

    mutation = ParameterMutation(
        strength=0.1,
        random_generator=random.Random(2),
    )

    reproduction = Reproduction(
        mutation=mutation,
        random_generator=random.Random(2),
        crossover=Crossover(
            strategy=GROUPED,
            random_generator=random.Random(2),
        ),
    )

    children = reproduction.create(
        population,
        count=50,
    )

    for child in children:

        for parameter in child.parameters:

            assert parameter.minimum <= parameter.value <= parameter.maximum
