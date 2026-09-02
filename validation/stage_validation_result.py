"""
validation/stage_validation_result.py

Result object for stage motion validation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageValidationResult:
    """
    Result of a stage motion validation.
    """

    valid: bool

    checked_steps: int

    failed_at_input_angle: float | None = None

    reason: str | None = None

    stage_id: int | None = None