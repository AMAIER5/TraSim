"""optimization/csv_parameter_factory.py

Creates optimization parameter templates
from a CSV-based MechanismDefinition.
"""

from __future__ import annotations

from model.mechanism_definition import (
    MechanismDefinition,
)
from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)

#: Default symmetric search-space offset around
#: each lever's CSV pivot value (millimetres).
DEFAULT_PIVOT_OFFSET = 20.0


class CsvParameterFactory:
    """
    Creates optimization parameters from
    lever definitions.

    Each optimizable lever property is mapped
    to a Parameter:

    - length (length_min..length_max)
    - angle (angle_min..angle_max)
    - pivot x / pivot y (symmetric search
      space around the CSV value; see
      ``pivot_bounds``)

    Levers with a ``pivot_on`` reference (mixer
    levers) have no pivot of their own: their
    pivot sits on the endpoint of the referenced
    lever, so no pivot parameter is created
    for them.
    """

    @staticmethod
    def create(
        definition: MechanismDefinition,
        *,
        pivot_bounds: (
            tuple[float, float] | None
        ) = None,
    ) -> ParameterSet:
        """
        Create a parameter template from
        a mechanism definition.

        Parameters
        ----------
        definition:
            The mechanism definition.
        pivot_bounds:
            Optional ``(minimum_offset, maximum_offset)``
            pair applied symmetrically around each
            lever's CSV pivot x/y value, e.g. ``(-20,
            20)`` for a +/-20 unit box.  When ``None``
            (default), the default search space
            ``DEFAULT_PIVOT_OFFSET`` is applied, so
            pivots are optimizable.  Pass ``(0, 0)``
            to nail pivots down at the CSV value.
        """
        if pivot_bounds is None:
            offset_min = -DEFAULT_PIVOT_OFFSET
            offset_max = DEFAULT_PIVOT_OFFSET
        else:
            offset_min, offset_max = pivot_bounds
            if offset_min > offset_max:
                raise ValueError(
                    "pivot_bounds minimum must not be "
                    "greater than maximum"
                )

        parameters = []
        for lever in definition.levers:
            parameters.append(
                Parameter(
                    name=f"lever.{lever.id}.length",
                    minimum=lever.length_min,
                    maximum=lever.length_max,
                    value=lever.length_start,
                )
            )
            parameters.append(
                Parameter(
                    name=f"lever.{lever.id}.angle",
                    minimum=lever.angle_min,
                    maximum=lever.angle_max,
                    value=lever.angle_start,
                )
            )
            if lever.pivot_on is not None:
                continue
            parameters.append(
                Parameter(
                    name=(
                        f"lever.{lever.id}.pivot.x"
                    ),
                    minimum=lever.pivot.x + offset_min,
                    maximum=lever.pivot.x + offset_max,
                    value=lever.pivot.x,
                )
            )
            parameters.append(
                Parameter(
                    name=(
                        f"lever.{lever.id}.pivot.y"
                    ),
                    minimum=lever.pivot.y + offset_min,
                    maximum=lever.pivot.y + offset_max,
                    value=lever.pivot.y,
                )
            )
        return ParameterSet(
            parameters=tuple(parameters),
        )
