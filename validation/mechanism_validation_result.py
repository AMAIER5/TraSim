"""
validation/mechanism_validation_result.py

Result object for complete mechanism validation.
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
        Return first failed stage index.
        """

        for stage in self.stages:
            if not stage.valid:
                return stage.stage_id

        return None