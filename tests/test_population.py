"""
tests/test_population.py

Tests for optimization populations.

Issue #21: Added tests verifying the return type of
``__iter__`` and that the type annotation is present.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest

from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population import (
    Population,
)


def create_parameter_set(
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


def test_population_creation():
    population = Population(
        (
            create_parameter_set(20.0),
            create_parameter_set(40.0),
        )
    )

    assert len(
        population
    ) == 2


def test_population_access():
    population = Population(
        (
            create_parameter_set(20.0),
        )
    )

    assert population[0].get(
        "length"
    ).value == 20.0


def test_population_is_iterable():
    population = Population(
        (
            create_parameter_set(20.0),
            create_parameter_set(40.0),
        )
    )

    values = [
        item.get("length").value
        for item in population
    ]

    assert values == [
        20.0,
        40.0,
    ]


def test_empty_population_is_rejected():

    with pytest.raises(
        ValueError
    ):
        Population(
            ()
        )


def test_population_is_immutable():
    population = Population(
        (
            create_parameter_set(20.0),
        )
    )

    with pytest.raises(
        AttributeError
    ):
        population.members = ()


# ---------------------------------------------------------------------------
# Issue #21: __iter__ return type annotation
# ---------------------------------------------------------------------------

def test_iter_has_return_annotation():
    """
    Issue #21: __iter__ must have a return type
    annotation (Iterator[ParameterSet]).
    """

    hints = Population.__iter__.__annotations__

    assert "return" in hints

    # The annotation should be Iterator[ParameterSet]
    # or compatible.
    return_hint = hints["return"]

    # Check that it's Iterator (either the typing or
    # collections.abc version).
    assert return_hint is not None


def test_iter_returns_iterator():
    """
    Issue #21: __iter__ must return an iterator object.
    """

    population = Population(
        (
            create_parameter_set(20.0),
            create_parameter_set(40.0),
        )
    )

    result = iter(population)

    assert hasattr(result, "__next__")
    assert hasattr(result, "__iter__")


def test_iter_yields_parameter_sets():
    """
    Issue #21: The iterator must yield ParameterSet
    instances, not something else.
    """

    population = Population(
        (
            create_parameter_set(20.0),
            create_parameter_set(40.0),
        )
    )

    for item in population:
        assert isinstance(item, ParameterSet)


def test_iter_signature_returns_iterator():
    """
    Issue #21: Verify via inspect that the return
    annotation resolves to Iterator[ParameterSet].
    """

    from optimization.population import Population as P

    sig = inspect.signature(P.__iter__)
    return_annotation = sig.return_annotation

    # The string representation should mention Iterator
    # and ParameterSet.
    assert "Iterator" in str(return_annotation)
    assert "ParameterSet" in str(return_annotation)