"""
optimization/mechanism_simulator.py

Adapter that simulates a complete mechanism.

The simulator executes every stage of a mechanism and
returns the resulting transfer curves.
"""

from __future__ import annotations

from typing import Callable

from mechanics.mechanism import Mechanism


class MechanismSimulator:
    """
    Simulates complete mechanisms.

    Each stage is simulated independently using the
    supplied stage simulator.
    """

    def __init__(
        self,
        *,
        stage_simulator: Callable,
    ) -> None:

        self._stage_simulator = (
            stage_simulator
        )

    def simulate(
        self,
        mechanism: Mechanism,
    ) -> tuple:
        """
        Simulate every stage.

        Returns
        -------
        tuple
            Simulation result for every stage.
        """

        return tuple(
            self._stage_simulator(stage)
            for stage in mechanism.stages
        )