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


class CsvParameterFactory:
    """
    Creates optimization parameters from
    lever definitions.

    Each optimizable lever property is mapped
    to a Parameter:

    - length (length_min..length_max)
    - angle (angle_min..angle_max)
    - pivot x / pivot y (fixed by default;
      see ``pivot_bounds``)

    Pivot parameters follow the same mechanism as
    length and angle: a parameter exists for every
    lever row, and a pivot can be nailed down (or
    freed for optimization) via ``minimum == maximum``
    fixed parameters, exactly like the existing
    length/angle parameters.
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
            (default), pivot parameters are fixed at
            the CSV value (``minimum == maximum``), so
            pivots stay nailed down unless explicitly
            freed.
        """
        if pivot_bounds is not None:
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
            if pivot_bounds is None:
                pivot_x_min = lever.pivot.x
                pivot_x_max = lever.pivot.x
                pivot_y_min = lever.pivot.y
                pivot_y_max = lever.pivot.y
            else:
                offset_min, offset_max = pivot_bounds
                pivot_x_min = (
                    lever.pivot.x + offset_min
                )
                pivot_x_max = (
                    lever.pivot.x + offset_max
                )
                pivot_y_min = (
                    lever.pivot.y + offset_min
                )
                pivot_y_max = (
                    lever.pivot.y + offset_max
                )
            parameters.append(
                Parameter(
                    name=(
                        f"lever.{lever.id}.pivot.x"
                    ),
                    minimum=pivot_x_min,
                    maximum=pivot_x_max,
                    value=lever.pivot.x,
                )
            )
            parameters.append(
                Parameter(
                    name=(
                        f"lever.{lever.id}.pivot.y"
                    ),
                    minimum=pivot_y_min,
                    maximum=pivot_y_max,
                    value=lever.pivot.y,
                )
            )
        return ParameterSet(
            parameters=tuple(parameters),
        )
