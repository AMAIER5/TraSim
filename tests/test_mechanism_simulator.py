"""
tests/test_mechanism_simulator.py

Tests for MechanismSimulator.
"""

from __future__ import annotations

from mechanics.mechanism import Mechanism
from mechanics.stage import Stage

from optimization.mechanism_simulator import (
    MechanismSimulator,
)


def create_stage() -> Stage:

    return Stage(
        input_lever=None,
        output_lever=None,
        rod_length=100.0,
        input_angle=0.0,
        output_angle=0.0,
        input_endpoint=None,
        output_endpoint=None,
    )


def test_simulates_single_stage():

    mechanism = Mechanism(
        stages=(
            create_stage(),
        )
    )

    simulator = MechanismSimulator(
        stage_simulator=lambda stage: "curve"
    )

    result = simulator.simulate(
        mechanism
    )

    assert result == (
        "curve",
    )


def test_simulates_all_stages():

    mechanism = Mechanism(
        stages=(
            create_stage(),
            create_stage(),
            create_stage(),
        )
    )

    simulator = MechanismSimulator(
        stage_simulator=lambda stage: id(stage)
    )

    result = simulator.simulate(
        mechanism
    )

    assert len(result) == 3

    assert len(set(result)) == 3


def test_stage_order_is_preserved():

    stage1 = create_stage()
    stage2 = create_stage()

    mechanism = Mechanism(
        stages=(
            stage1,
            stage2,
        )
    )

    simulator = MechanismSimulator(
        stage_simulator=lambda stage: (
            "A"
            if stage is stage1
            else "B"
        )
    )

    result = simulator.simulate(
        mechanism
    )

    assert result == (
        "A",
        "B",
    )