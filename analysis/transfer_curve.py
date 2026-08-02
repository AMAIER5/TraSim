"""
analysis/transfer_curve.py

Representation of an input/output angle relationship.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TransferCurve:
    """
    Represents a kinematic transfer function.

    Input and output angles must be ordered
    and have equal length.

    The input angles may be either ascending
    or descending.
    """

    input_angles: tuple[float, ...]

    output_angles: tuple[float, ...]

    _ascending: bool = field(
        init=False,
        repr=False,
    )

    _minimum: float = field(
        init=False,
        repr=False,
    )

    _maximum: float = field(
        init=False,
        repr=False,
    )

    _lookup_inputs: tuple[float, ...] = field(
        init=False,
        repr=False,
    )

    _lookup_outputs: tuple[float, ...] = field(
        init=False,
        repr=False,
    )


    def __post_init__(self) -> None:

        if len(
            self.input_angles
        ) != len(
            self.output_angles
        ):
            raise ValueError(
                "input and output length mismatch"
            )

        if len(
            self.input_angles
        ) < 2:
            raise ValueError(
                "at least two points required"
            )


        ascending = (
            self.input_angles[0]
            <
            self.input_angles[-1]
        )

        object.__setattr__(
            self,
            "_ascending",
            ascending,
        )


        object.__setattr__(
            self,
            "_minimum",
            min(self.input_angles),
        )

        object.__setattr__(
            self,
            "_maximum",
            max(self.input_angles),
        )


        if ascending:

            object.__setattr__(
                self,
                "_lookup_inputs",
                self.input_angles,
            )

            object.__setattr__(
                self,
                "_lookup_outputs",
                self.output_angles,
            )

        else:

            object.__setattr__(
                self,
                "_lookup_inputs",
                tuple(
                    reversed(
                        self.input_angles
                    )
                ),
            )

            object.__setattr__(
                self,
                "_lookup_outputs",
                tuple(
                    reversed(
                        self.output_angles
                    )
                ),
            )


    def output_at(
        self,
        input_angle: float,
    ) -> float:
        """
        Linear interpolation of output angle.

        The input range must be covered by
        the curve.

        Supports both ascending and descending
        input angle sequences.
        """

        eps = 1e-12

        if (
            input_angle
            <
            self._minimum - eps
            or
            input_angle
            >
            self._maximum + eps
        ):
            raise ValueError(
                "input angle outside curve range"
            )


        input_angle = min(
            max(
                input_angle,
                self._minimum,
            ),
            self._maximum,
        )


        index = bisect_left(
            self._lookup_inputs,
            input_angle,
        )


        if index == 0:

            return self._lookup_outputs[0]


        if index >= len(
            self._lookup_inputs
        ):

            return self._lookup_outputs[-1]


        x0 = self._lookup_inputs[
            index - 1
        ]

        x1 = self._lookup_inputs[
            index
        ]

        y0 = self._lookup_outputs[
            index - 1
        ]

        y1 = self._lookup_outputs[
            index
        ]


        factor = (
            input_angle - x0
        ) / (
            x1 - x0
        )

        return (
            y0
            +
            factor
            *
            (
                y1
                -
                y0
            )
        )