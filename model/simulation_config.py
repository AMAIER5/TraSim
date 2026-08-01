from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """
    Global optimization and simulation parameters.
    """

    population_size: int
    children_per_generation: int
    generations: int

    target_error: float

    mutation_rate: float
    elite_size: int

    motion_start: float
    motion_end: float
    motion_step: float