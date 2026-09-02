"""
simulation/simulation_result.py

Container for simulation output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """
    Result of a simulation run.

    All angles are stored in radians.
    """

    input_angles: tuple[float, ...]

    output_angles: tuple[float, ...]

    success: bool

    blocked_at: float | None = None
    
    def __post_init__(self):

        if len(self.input_angles) != len(self.output_angles):

            raise ValueError(
                "Input and output angle count must match."
            )