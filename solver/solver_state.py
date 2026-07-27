"""
solver/solver_state.py

Persistent solver state.

The solver follows one continuous branch of motion.

The state stores the previously calculated solution so that the
next iteration can search locally instead of starting from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverState:
    """
    Solver state between two simulation steps.

    Parameters
    ----------
    last_input_angle:
        Previously solved input angle [rad].

    last_output_angle:
        Previously solved output angle [rad].

    direction:
        Simulation direction.

            +1 = increasing input angle
            -1 = decreasing input angle
    """

    last_input_angle: float
    last_output_angle: float
    direction: int = 1

    def __post_init__(self) -> None:
        """
        Validate solver state.
        """

        if self.direction not in (-1, 1):
            raise ValueError(
                "direction must be either +1 or -1."
            )

    def next(
        self,
        *,
        input_angle: float,
        output_angle: float,
    ) -> SolverState:
        """
        Return the next immutable solver state.

        Parameters
        ----------
        input_angle:
            Newly solved input angle.

        output_angle:
            Newly solved output angle.
        """

        return SolverState(
            last_input_angle=input_angle,
            last_output_angle=output_angle,
            direction=self.direction,
        )

    def reversed(self) -> SolverState:
        """
        Return a new solver state with reversed simulation direction.
        """

        return SolverState(
            last_input_angle=self.last_input_angle,
            last_output_angle=self.last_output_angle,
            direction=-self.direction,
        )