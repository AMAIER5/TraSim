from __future__ import annotations

import pytest

from core.vector3d import Vector3D
from model.lever_definition import LeverDefinition
from model.mechanism_definition import MechanismDefinition


def create_lever(
    lever_id: int,
    *,
    driver: int | None = None,
    coupled: int | None = None,
) -> LeverDefinition:
    """
    Create a minimal lever definition for testing.
    """

    return LeverDefinition(
        id=lever_id,
        length_min=40,
        length_max=100,
        length_start=60,
        angle_min=-45,
        angle_max=45,
        angle_start=0,
        axis=Vector3D(0, 0, 1),
        driver=driver,
        coupled=coupled,
    )


def test_lever_count():
    mechanism = MechanismDefinition(
        (
            create_lever(1),
            create_lever(2, driver=1),
            create_lever(3, driver=2),
        )
    )

    assert mechanism.lever_count == 3


def test_get_lever():
    mechanism = MechanismDefinition(
        (
            create_lever(1),
            create_lever(2, driver=1),
        )
    )

    lever = mechanism.get_lever(2)

    assert lever.id == 2
    assert lever.driver == 1


def test_get_unknown_lever_raises_error():
    mechanism = MechanismDefinition(
        (
            create_lever(1),
        )
    )

    with pytest.raises(KeyError):
        mechanism.get_lever(99)


def test_input_lever():
    mechanism = MechanismDefinition(
        (
            create_lever(1),
            create_lever(2, driver=1),
        )
    )

    assert mechanism.input_lever.id == 1


def test_coupled_levers():
    mechanism = MechanismDefinition(
        (
            create_lever(1),
            create_lever(2, driver=1),
            create_lever(3, coupled=2),
        )
    )

    coupled = mechanism.coupled_levers

    assert len(coupled) == 1
    assert coupled[0].id == 3


def test_driven_levers():
    mechanism = MechanismDefinition(
        (
            create_lever(1),
            create_lever(2, driver=1),
            create_lever(3, driver=2),
        )
    )

    driven = mechanism.driven_levers

    assert len(driven) == 2
    assert driven[0].id == 2
    assert driven[1].id == 3


def test_duplicate_lever_ids_raise_error():
    with pytest.raises(ValueError, match="Lever IDs must be unique"):
        MechanismDefinition(
            (
                create_lever(1),
                create_lever(1),
            )
        )
        
def test_empty_mechanism_has_no_input_lever():
    mechanism = MechanismDefinition(())

    with pytest.raises(ValueError, match="Mechanism contains no levers"):
        _ = mechanism.input_lever