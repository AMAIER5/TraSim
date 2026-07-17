"""
solver/mechanism_state.py

State container for complete mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

from solver.solver_state import SolverState


@dataclass(frozen=True, slots=True)
class MechanismState:
    """
    Stores the solver state of all mechanism stages.

    The order of stage_states must match
    the order of stages in the Mechanism.
    """

    stage_states: tuple[SolverState, ...]

    def __post_init__(self) -> None:

        if not isinstance(
            self.stage_states,
            tuple,
        ):

            raise TypeError(
                "stage_states must be a tuple"
            )

    def stage_count(self) -> int:
        """
        Number of stored stage states.
        """

        return len(
            self.stage_states
        )