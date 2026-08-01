"""
solver/solver_state.py

Persistent solver state.

Stores the previous solution and local motion information
to continue the physical motion branch.

The state is intentionally independent from simulation timing
or motion range handling. It only describes the local kinematic
branch state of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SolverState:
    """
    Persistent state of a kinematic solver.

    The solver follows one continuous physical branch.
    The state stores the previous solution and an estimate
    of the local output velocity.

    Angles are stored in radians.

    Parameters
    ----------
    last_input_angle:
        Previous solved input angle.

    last_output_angle:
        Previous solved output angle.

    direction:
        Motion direction of the input angle.

        +1:
            increasing input angle

        -1:
            decreasing input angle

        0:
            undefined initial direction

    output_velocity:
        Local approximation:

            d(output_angle) / d(input_angle)

        Used to predict the next output angle.
    """

    last_input_angle: float
    last_output_angle: float

    direction: int = 0

    output_velocity: float = 0.0

    def __post_init__(self) -> None:

        if self.direction not in (-1, 0, 1):
            raise ValueError(
                "direction must be -1, 0 or 1."
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_output(
        self,
        input_angle: float,
    ) -> float:
        """
        Predict output angle for a new input angle.

        Uses linear extrapolation based on the previous
        local velocity estimate.
        """

        delta_input = (
            input_angle
            -
            self.last_input_angle
        )

        return (
            self.last_output_angle
            +
            self.output_velocity
            *
            delta_input
        )

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def next(
        self,
        *,
        input_angle: float,
        output_angle: float,
    ) -> SolverState:
        """
        Create the next solver state after a successful solution.

        Updates the local velocity estimate from the latest
        motion segment.
        """

        delta_input = (
            input_angle
            -
            self.last_input_angle
        )

        if abs(delta_input) > 1e-12:

            velocity = (
                output_angle
                -
                self.last_output_angle
            ) / delta_input

        else:

            velocity = self.output_velocity


        direction = self.direction


        if delta_input > 0:

            direction = 1

        elif delta_input < 0:

            direction = -1


        return SolverState(
            last_input_angle=input_angle,
            last_output_angle=output_angle,
            direction=direction,
            output_velocity=velocity,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def initial(
        cls,
        *,
        input_angle: float,
        output_angle: float,
    ) -> SolverState:
        """
        Create an initial solver state.

        The initial direction is unknown until the first
        movement step is completed.
        """

        return cls(
            last_input_angle=input_angle,
            last_output_angle=output_angle,
            direction=0,
            output_velocity=0.0,
        )

    # ------------------------------------------------------------------
    # Direction handling
    # ------------------------------------------------------------------

    def reversed(self) -> SolverState:
        """
        Create a state for reversed motion.

        The stored geometry remains identical.
        Only the expected motion direction changes.
        """

        return SolverState(
            last_input_angle=self.last_input_angle,
            last_output_angle=self.last_output_angle,
            direction=-self.direction,
            output_velocity=self.output_velocity,
        )