"""
tests/test_solver_state.py

Unit tests for SolverState.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

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
    assert state.direction == 1


# ---------------------------------------------------------------------------
# Explicit direction
# ---------------------------------------------------------------------------

def test_solver_state_backward_direction():

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
            direction=0,
        )


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