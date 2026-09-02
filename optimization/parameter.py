"""
optimization/parameter.py

Definition of a single optimization parameter.

Issue #19: The original code rejected ``minimum == maximum``
with ``ValueError("minimum must be smaller than maximum")``.
This prevents fixed (non-optimizable) parameters such as
``Parameter(min=50, max=50, value=50)``, which are needed
for CSV fixtures with fixed levers (e.g.
``length_min=length_max=50``).

The fix allows ``minimum == maximum`` as a degenerate
"fixed" parameter.  Only ``minimum > maximum`` is now
rejected.  When ``minimum == maximum``, the value must
equal that same number.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Parameter:
    """
    A bounded optimization parameter.

    The optimizer may vary the value only
    inside the defined range.

    When ``minimum == maximum``, the parameter is fixed
    (non-optimizable) and ``value`` must equal that same
    number.
    """

    name: str

    minimum: float

    maximum: float

    value: float

    def __post_init__(self) -> None:

        # Issue #19: Allow minimum == maximum (fixed parameter).
        # Only minimum > maximum is invalid.
        if self.minimum > self.maximum:
            raise ValueError(
                "minimum must not be greater than maximum"
            )

        if not (
            self.minimum
            <=
            self.value
            <=
            self.maximum
        ):
            raise ValueError(
                "value outside parameter range"
            )

        if not self.name:
            raise ValueError(
                "parameter name must not be empty"
            )

    @property
    def is_fixed(self) -> bool:
        """
        Issue #19: A parameter is fixed when its range
        is a single point (minimum == maximum).
        """

        return self.minimum == self.maximum

    @property
    def range(self) -> float:
        """
        Issue #19: The optimizable range.  Zero for
        fixed parameters.
        """

        return self.maximum - self.minimum