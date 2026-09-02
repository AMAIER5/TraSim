"""
validation/mechanism_motion_validator.py

Validation of complete mechanisms over their motion range.

Issue #6: There were two MechanismValidationResult classes:
  1. A hand-written class in this file (used by the validator)
  2. A dataclass in mechanism_validation_result.py (unused,
     with a different API — failed_stage property)

The dataclass version (mechanism_validation_result.py) has been
deleted.  MechanismValidationResult now lives in its own module
(validation/mechanism_validation_result.py) as a frozen dataclass
that is imported by both the validator and any code that needs
the result type.  The hand-written class that was previously
inlined here has been removed.
"""

from __future__ import annotations

from mechanics.mechanism import Mechanism
from simulation.motion_range import MotionRange
from validation.mechanism_validation_result import (
    MechanismValidationResult,
)
from validation.stage_motion_validator import (
    StageMotionValidator,
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
            stages=tuple(results),
        )