"""
validation/mechanism_validation_result.py

Result object for complete mechanism validation.

Issue #6: Previously there were two MechanismValidationResult
classes — a hand-written one inlined in
mechanism_motion_validator.py and a dataclass here with a
different API (failed_stage property).  The hand-written
class has been removed and this dataclass is now the single
source of truth.

This dataclass preserves the API used by existing tests:
  - .stages  (tuple of StageValidationResult)
  - .valid   (bool, True if all stages valid)

The failed_stage property from the old version is retained
for completeness but was never used by the validator.
"""

from __future__ import annotations

from dataclasses import dataclass

from validation.stage_validation_result import (
    StageValidationResult,
)


@dataclass(frozen=True, slots=True)
class MechanismValidationResult:
    """
    Result of a complete mechanism validation.

    Parameters
    ----------
    stages:
        Validation results for each stage, in stage order.

    Properties
    ----------
    valid:
        True if all stages are valid.

    failed_stage:
        Stage ID of the first failed stage, or None.
    """

    stages: tuple[StageValidationResult, ...]

    @property
    def valid(self) -> bool:
        """
        True if all stages are valid.
        """

        return all(
            stage.valid
            for stage in self.stages
        )

    @property
    def failed_stage(self) -> int | None:
        """
        Return first failed stage ID.
        """

        for stage in self.stages:
            if not stage.valid:
                return stage.stage_id

        return None