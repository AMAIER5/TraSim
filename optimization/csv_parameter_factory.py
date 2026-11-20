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

#: Default symmetric room around the unbraked pair pivot
#: within which the pivot may be displaced for the braked
#: state (millimetres).
DEFAULT_SHIFT_OFFSET = 20.0


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
            # Per-axis optional pivot limits from the
            # mechanism definition take precedence over the
            # symmetric offset: an axis with explicit limits
            # keeps them, unrestricted axes fall back to the
            # offset around the CSV value.
            (
                (x_min, x_max),
                (y_min, y_max),
                (z_min, z_max),
            ) = lever.pivot_limits
            parameters.append(
                Parameter(
                    name=(
                        f"lever.{lever.id}.pivot.x"
                    ),
                    minimum=(
                        x_min
                        if x_min is not None
                        else lever.pivot.x
                        + offset_min
                    ),
                    maximum=(
                        x_max
                        if x_max is not None
                        else lever.pivot.x
                        + offset_max
                    ),
                    value=lever.pivot.x,
                )
            )
            parameters.append(
                Parameter(
                    name=(
                        f"lever.{lever.id}.pivot.y"
                    ),
                    minimum=(
                        y_min
                        if y_min is not None
                        else lever.pivot.y
                        + offset_min
                    ),
                    maximum=(
                        y_max
                        if y_max is not None
                        else lever.pivot.y
                        + offset_max
                    ),
                    value=lever.pivot.y,
                )
            )
            parameters.append(
                Parameter(
                    name=(
                        f"lever.{lever.id}.pivot.z"
                    ),
                    minimum=(
                        z_min
                        if z_min is not None
                        else lever.pivot.z
                        + offset_min
                    ),
                    maximum=(
                        z_max
                        if z_max is not None
                        else lever.pivot.z
                        + offset_max
                    ),
                    value=lever.pivot.z,
                )
            )
        return ParameterSet(
            parameters=tuple(parameters),
        )

    @staticmethod
    def create_pivot_shift(
        definition: MechanismDefinition,
        *,
        shift_lever_id: int,
        pivot_bounds: (
            tuple[float, float] | None
        ) = None,
        shift_bounds: (
            tuple[float, float] | None
        ) = None,
    ) -> ParameterSet:
        """Create a parameter template for the pivot-shift
        approach: the standard lever parameters plus the two
        displacement parameters of the shifted pair pivot.

        The lever pair shares ONE common pivot: the partner
        levers get no pivot parameters of their own; their
        pivot always follows the shift lever's pivot, so the
        pair cannot drift apart during optimization.

        The shift parameters bound the ROOM within which the
        common pivot of the lever pair may be displaced for
        the braked state.  Their default value is 0 (the
        unbraked position).  The room follows the shift
        lever's optional per-axis pivot limits from the
        mechanism definition (``pivot_x_min`` .. ``pivot_z_max``
        columns): a displaced pivot stays inside the same
        admissible room in which the (unbraked) pivot itself
        may be optimized, e.g. ``shift.x`` ranges over
        ``[pivot_x_min - pivot_x, pivot_x_max - pivot_x]``.
        Without defined limits, ``shift_bounds`` configures
        the room as a symmetric ``(minimum_offset,
        maximum_offset)`` box around 0, defaulting to
        ``DEFAULT_SHIFT_OFFSET``; an explicit ``shift_bounds``
        also overrides the definition limits.
        """
        leader = definition.get_lever(shift_lever_id)
        pair_ids = {
            lever.id
            for lever in definition.levers
            if (
                lever.pivot.x == leader.pivot.x
                and lever.pivot.y == leader.pivot.y
                and lever.pivot.z == leader.pivot.z
            )
        }
        pair_ids.discard(shift_lever_id)
        (
            (x_min, x_max),
            (y_min, y_max),
            _z_limits,
        ) = leader.pivot_limits
        default_x_bounds = (
            (
                x_min - leader.pivot.x
                if x_min is not None
                else -DEFAULT_SHIFT_OFFSET
            ),
            x_max - leader.pivot.x
            if x_max is not None
            else DEFAULT_SHIFT_OFFSET,
        )
        default_y_bounds = (
            (
                y_min - leader.pivot.y
                if y_min is not None
                else -DEFAULT_SHIFT_OFFSET
            ),
            y_max - leader.pivot.y
            if y_max is not None
            else DEFAULT_SHIFT_OFFSET,
        )
        if shift_bounds is None:
            shift_x_bounds = default_x_bounds
            shift_y_bounds = default_y_bounds
        else:
            shift_min, shift_max = shift_bounds
            if shift_min > shift_max:
                raise ValueError(
                    "shift_bounds minimum must not be "
                    "greater than maximum"
                )
            shift_x_bounds = (shift_min, shift_max)
            shift_y_bounds = (shift_min, shift_max)
        template = CsvParameterFactory.create(
            definition,
            pivot_bounds=pivot_bounds,
        )
        parameters = [
            parameter
            for parameter in template.parameters
            if not any(
                parameter.name
                == (
                    f"lever.{lever_id}.pivot.{axis}"
                )
                for lever_id in pair_ids
                for axis in ("x", "y", "z")
            )
        ]
        for name, bounds in (
            (
                f"lever.{shift_lever_id}.shift.x",
                shift_x_bounds,
            ),
            (
                f"lever.{shift_lever_id}.shift.y",
                shift_y_bounds,
            ),
        ):
            parameters.append(
                Parameter(
                    name=name,
                    minimum=bounds[0],
                    maximum=bounds[1],
                    value=0.0,
                )
            )
        return ParameterSet(
            parameters=tuple(parameters),
        )
