"""
tests/test_solver_state.py

Unit tests for SolverState.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from solver.solver_state import SolverState


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_solver_state_creation():

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=1.0,
    )

    assert state.last_input_angle == 0.0
    assert state.last_output_angle == 1.0
    assert state.direction == 0
    assert state.output_velocity == 0.0



# ---------------------------------------------------------------------------
# Explicit direction
# ---------------------------------------------------------------------------

def test_solver_state_explicit_forward_direction():

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
        direction=1,
    )

    assert state.direction == 1



def test_solver_state_explicit_backward_direction():

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
        direction=-1,
    )

    assert state.direction == -1



# ---------------------------------------------------------------------------
# Invalid direction
# ---------------------------------------------------------------------------

def test_solver_state_invalid_direction():

    with pytest.raises(ValueError):

        SolverState(
            last_input_angle=0.0,
            last_output_angle=0.0,
            direction=2,
        )



# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_solver_state_predict_output():

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=10.0,
        output_velocity=2.0,
    )

    predicted = state.predict_output(
        5.0,
    )

    assert predicted == 20.0



def test_solver_state_predict_output_without_velocity():

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=10.0,
    )

    predicted = state.predict_output(
        5.0,
    )

    assert predicted == 10.0



# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------

def test_solver_state_next_updates_velocity():

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    new_state = state.next(
        input_angle=1.0,
        output_angle=2.0,
    )

    assert new_state.last_input_angle == 1.0
    assert new_state.last_output_angle == 2.0
    assert new_state.direction == 1

    assert math.isclose(
        new_state.output_velocity,
        2.0,
    )



def test_solver_state_next_backward_motion():

    state = SolverState(
        last_input_angle=1.0,
        last_output_angle=2.0,
        direction=1,
        output_velocity=1.0,
    )

    new_state = state.next(
        input_angle=0.0,
        output_angle=1.0,
    )

    assert new_state.direction == -1



# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------

def test_solver_state_reversed():

    state = SolverState(
        last_input_angle=1.0,
        last_output_angle=2.0,
        direction=1,
        output_velocity=3.0,
    )

    reversed_state = state.reversed()

    assert reversed_state.last_input_angle == 1.0
    assert reversed_state.last_output_angle == 2.0
    assert reversed_state.direction == -1
    assert reversed_state.output_velocity == 3.0



# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

def test_solver_state_is_immutable():

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    with pytest.raises(FrozenInstanceError):

        state.direction = -1