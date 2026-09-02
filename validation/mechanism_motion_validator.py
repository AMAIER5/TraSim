"""
validation/mechanism_motion_validator.py

Validation of complete mechanisms over their motion range.
"""

from __future__ import annotations

from mechanics.mechanism import Mechanism
from simulation.motion_range import MotionRange
from validation.stage_motion_validator import (
    StageMotionValidator,
)
from validation.stage_validation_result import (
    StageValidationResult,
)


class MechanismValidationResult:
    """
    Container for complete mechanism validation.
    """

    def __init__(
        self,
        stages: tuple[StageValidationResult, ...],
    ) -> None:

        self.stages = stages

    @property
    def valid(self) -> bool:
        return all(
            stage.valid
            for stage in self.stages
        )


class MechanismMotionValidator:
    """
    Validate all stages of a mechanism.
    """

    def __init__(
        self,
        *,
        stage_validator: StageMotionValidator | None = None,
    ) -> None:

        self._stage_validator = (
            stage_validator
            if stage_validator is not None
            else StageMotionValidator()
        )


    def validate(
        self,
        mechanism: Mechanism,
        motion: MotionRange | None = None,
    ) -> MechanismValidationResult:
        """
        Validate all stages.

        If a motion range is supplied, it is converted into
        explicit input positions. Otherwise every stage's
        defined input range is checked.
        """

        results = []

        for index, stage in enumerate(
            mechanism.stages,
        ):

            if motion is None:

                result = (
                    self._stage_validator.validate(
                        stage,
                        stage_id=index,
                    )
                )

            else:

                result = (
                    self._stage_validator.validate_motion(
                        stage,
                        tuple(motion),
                        stage_id=index,
                    )
                )

            results.append(result)


        return MechanismValidationResult(
            tuple(results)
        )