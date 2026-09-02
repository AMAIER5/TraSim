from __future__ import annotations

import math

from mechanics.stage import Stage
from solver.stage_solver import StageSolver
from validation.stage_validation_result import (
    StageValidationResult,
)


class StageMotionValidator:
    """
    Validate whether a stage can follow its complete
    defined input motion range.
    """

    def __init__(
        self,
        *,
        steps: int = 50,
    ) -> None:

        self.steps = steps


    def validate(
        self,
        stage: Stage,
        *,
        stage_id=None,
    ) -> StageValidationResult:

        return self.validate_motion(
            stage,
            self._input_angles(stage),
            stage_id=stage_id,
        )


    def _input_angles(
        self,
        stage: Stage,
    ):
        """
        Generate test positions.
        """

        if not math.isfinite(
            stage.input_angle_min
        ) or not math.isfinite(
            stage.input_angle_max
        ):
            raise ValueError(
                "Stage input range must be finite"
            )

        step = (
            stage.input_angle_max
            -
            stage.input_angle_min
        ) / self.steps

        for i in range(
            self.steps + 1
        ):
            yield (
                stage.input_angle_min
                +
                i * step
            )


    def validate_motion(
        self,
        stage: Stage,
        input_angles,
        *,
        stage_id=None,
    ) -> StageValidationResult:
        """
        Validate a supplied sequence of input angles.
        """

        solver = StageSolver(stage)

        checked = 0

        for input_angle in input_angles:

            result = solver.solve(
                input_angle=input_angle,
            )

            checked += 1

            # No mathematical solution found
            if not result.success:

                return StageValidationResult(
                    valid=False,
                    checked_steps=checked,
                    failed_at_input_angle=input_angle,
                    reason=result.reason,
                    stage_id=stage_id,
                )

            # Mathematical solution exists,
            # but mechanical output range is violated
            if not stage.accepts_output_angle(
                result.angle
            ):

                return StageValidationResult(
                    valid=False,
                    checked_steps=checked,
                    failed_at_input_angle=input_angle,
                    reason="output_angle_limit",
                    stage_id=stage_id,
                )

        return StageValidationResult(
            valid=True,
            checked_steps=checked,
            stage_id=stage_id,
        )