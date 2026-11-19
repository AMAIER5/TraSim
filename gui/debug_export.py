"""Debug export of an optimization run as JSON.

The GUI offers a "Debug-Datei herunterladen" button beneath
the result HTML download.  The exported file contains
everything needed to reproduce or debug an unwanted result:
the initial mechanism definition, the optimized mechanism,
the target curve(s), all optimization settings, the manual
settings (per-point weights, optimizable parameters with their
bounds) and the run outcome (stop reason, fitness history,
best parameter values).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from analysis.target_curve import TargetCurve
from model.mechanism_definition import (
    MechanismDefinition,
)
from optimization.parameter_set import ParameterSet


def _deg(value: float) -> float:
    return math.degrees(value)


def _point(point: Any) -> dict[str, float]:
    return {
        "x": point.x,
        "y": point.y,
        "z": point.z,
    }


def _lever_dict(lever: Any) -> dict[str, Any]:
    return {
        "id": lever.id,
        "pivot": _point(lever.pivot),
        "length_min": lever.length_min,
        "length_max": lever.length_max,
        "length_start": lever.length_start,
        "angle_min_deg": _deg(lever.angle_min),
        "angle_max_deg": _deg(lever.angle_max),
        "angle_start_deg": _deg(lever.angle_start),
        "axis": _point(lever.axis),
        "reference_direction": (
            None
            if lever.reference_direction is None
            else _point(lever.reference_direction)
        ),
        "driver": lever.driver,
        "coupled": lever.coupled,
        "pivot_on": lever.pivot_on,
        # Issue #127: separate fully braked position of a
        # brake lever (None = legacy angle_max fallback).
        "brake_position_deg": (
            None
            if getattr(lever, "brake_position", None)
            is None
            else _deg(lever.brake_position)
        ),
    }


def definition_dict(
    definition: MechanismDefinition,
) -> dict[str, Any]:
    """Plain-dict representation of a mechanism definition
    (angles in degrees)."""
    return {
        "levers": [
            _lever_dict(lever)
            for lever in definition.levers
        ],
    }


def _parameter_dict(parameter: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": parameter.name,
        "minimum": parameter.minimum,
        "maximum": parameter.maximum,
        "value": parameter.value,
        "fixed": parameter.minimum == parameter.maximum,
    }
    if parameter.name.endswith(".angle"):
        entry["minimum_deg"] = _deg(parameter.minimum)
        entry["maximum_deg"] = _deg(parameter.maximum)
        entry["value_deg"] = _deg(parameter.value)
    return entry


def parameters_dict(
    template: ParameterSet,
) -> list[dict[str, Any]]:
    """Optimization parameters (the manual optimizable
    settings) with their bounds; angles also in degrees."""
    return [
        _parameter_dict(parameter)
        for parameter in template.parameters
    ]


def target_curve_dict(
    name: str,
    curve: TargetCurve,
    input_angles: Sequence[float],
) -> dict[str, Any]:
    """One target curve sampled at its support points
    (angles in degrees), including per-point weights."""
    weights = getattr(curve, "weights", None)
    return {
        "name": name,
        "input_angles_deg": [
            _deg(angle) for angle in input_angles
        ],
        "output_angles_deg": [
            _deg(curve.evaluate(angle))
            for angle in input_angles
        ],
        "weights": (
            None
            if weights is None
            else [float(weight) for weight in weights]
        ),
    }


def _values_dict(
    values: Mapping[str, float],
) -> dict[str, Any]:
    """Best parameter values; angles additionally in
    degrees."""
    result: dict[str, Any] = {}
    for name, value in values.items():
        result[name] = value
        if name.endswith(".angle"):
            result[name + "_deg"] = _deg(value)
    return result


def build_debug_dict(
    *,
    mode: str,
    initial_definition: MechanismDefinition,
    target_curves: Sequence[dict[str, Any]],
    optimization_settings: Mapping[str, Any],
    parameters: Sequence[dict[str, Any]],
    result: Mapping[str, Any],
    optimized_definition: (
        MechanismDefinition | None
    ) = None,
    initial_mechanism_csv: str = "",
    optimized_mechanism_csv: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete debug export as a plain dict."""
    export: dict[str, Any] = {
        "format": "TraSim debug export",
        "version": 1,
        "created_utc": datetime.now(
            UTC
        ).isoformat(),
        "mode": mode,
        "initial_mechanism": definition_dict(
            initial_definition
        ),
        "target_curves": list(target_curves),
        "optimization_settings": dict(
            optimization_settings
        ),
        "optimizable_parameters": list(parameters),
        "result": dict(result),
    }
    if initial_mechanism_csv:
        export["initial_mechanism_csv"] = (
            initial_mechanism_csv
        )
    if optimized_definition is not None:
        export["optimized_mechanism"] = definition_dict(
            optimized_definition
        )
    if optimized_mechanism_csv:
        export["optimized_mechanism_csv"] = (
            optimized_mechanism_csv
        )
    if extra:
        export["extra"] = dict(extra)
    return export


def debug_result_info(result: Any) -> dict[str, Any]:
    """Run outcome fields shared by the single and mixed
    result objects."""
    info: dict[str, Any] = {
        "stop_reason": result.stop_reason,
        "best_score": result.best_score,
        "generations": result.generations,
        "fitness_history": [
            float(value)
            for value in result.fitness_history
        ],
    }
    best_values = getattr(result, "best_values", None)
    if best_values:
        info["best_values"] = _values_dict(best_values)
    brake_positions_deg = getattr(
        result, "brake_positions_deg", None
    )
    if brake_positions_deg:
        info["brake_positions_deg"] = [
            float(value)
            for value in brake_positions_deg
        ]
    min_transmission = getattr(
        result, "min_transmission_angle_deg", None
    )
    if min_transmission is not None:
        info["min_transmission_angle_deg"] = (
            float(min_transmission)
        )
    return info


def debug_export_json(**kwargs: Any) -> str:
    """Serialize the debug export to a pretty JSON
    string."""
    return json.dumps(
        build_debug_dict(**kwargs),
        indent=2,
        ensure_ascii=False,
    )
