"""
tests/test_mechanism_state.py

Tests for MechanismState.
"""

from __future__ import annotations

import pytest

from solver.mechanism_state import MechanismState
from solver.solver_state import SolverState


def create_state(
    angle: float,
) -> SolverState:

    return SolverState(
        last_input_angle=angle,
        last_output_angle=angle,
    )


def test_empty_mechanism_state():

    state = MechanismState(
        stage_states=()
    )

    assert len(
        state.stage_states
    ) == 0


def test_mechanism_state_stores_stage_states():

    stage_a = create_state(0.0)

    stage_b = create_state(1.0)

    mechanism_state = MechanismState(
        stage_states=(
            stage_a,
            stage_b,
        )
    )

    assert len(
        mechanism_state.stage_states
    ) == 2

    assert (
        mechanism_state.stage_states[0]
        ==
        stage_a
    )

    assert (
        mechanism_state.stage_states[1]
        ==
        stage_b
    )


def test_mechanism_state_is_immutable():

    state = MechanismState(
        stage_states=()
    )

    with pytest.raises(
        AttributeError
    ):

        state.stage_states = ()


def test_mechanism_state_requires_tuple():

    with pytest.raises(
        TypeError
    ):

        MechanismState(
            stage_states=[]
        )