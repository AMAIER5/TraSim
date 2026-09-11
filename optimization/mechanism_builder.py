"""
optimization/mechanism_builder.py

Protocol for mechanism builders used by the optimization workflow.
"""

from __future__ import annotations

from typing import Protocol

from mechanics.mechanism import Mechanism
from optimization.parameter_set import ParameterSet
from validation.stage_validation_result import (
    StageValidationResult,
)


class MechanismBuilder(Protocol):
    """
    Builds a mechanism from a set of optimization parameters.
    """

    def build(self, parameters: ParameterSet) -> Mechanism:
        """
        Create and return a mechanism using the supplied parameter values.
        """
        ...

    def get_validation_results(
        self,
    ) -> tuple[StageValidationResult, ...]:
        """
        Return the stage validation results from the most recent build.

        Builders that do not validate return an empty tuple.
        """
        ...