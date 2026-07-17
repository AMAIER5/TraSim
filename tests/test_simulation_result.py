"""
tests/test_simulation_result.py

Tests for SimulationResult.
"""

from __future__ import annotations

import math

import pytest

from simulation.simulation_result import SimulationResult


def test_simulation_result_creation():

    result = SimulationResult(
        input_angles=(0.0, 1.0),
        output_angles=(0.0, 0.5),
        success=True,
    )

    assert len(result.input_angles) == 2
    assert len(result.output_angles) == 2
    assert result.success is True
    assert result.blocked_at is None


def test_simulation_result_blocked():

    result = SimulationResult(
        input_angles=(0.0, 0.5),
        output_angles=(0.0, 0.2),
        success=False,
        blocked_at=1.0,
    )

    assert result.success is False

    assert math.isclose(
        result.blocked_at,
        1.0,
    )


def test_simulation_result_is_immutable():

    result = SimulationResult(
        input_angles=(),
        output_angles=(),
        success=True,
    )

    with pytest.raises(
        AttributeError
    ):

        result.success = False