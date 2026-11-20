"""tests/test_pivot_shift_gui_workflow.py

Tests for the GUI workflow of the pivot shift mode
(gui/pivot_shift_workflow.py): input loading and
validation, the parameter template, a short
optimization run and accessibility guards of the
workflow layer.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from analysis.target_curve import TargetCurve
from gui.pivot_shift_workflow import (
    PivotShiftSettings,
    load_pivot_shift_inputs,
    pivot_pair_candidates,
    pivot_shift_inputs_from_parts,
    pivot_shift_parameter_template,
    run_pivot_shift_optimization,
)
from gui.workflow import WorkflowError
from mechanism_io.csv_reader import CsvReader
from model.mechanism_definition import (
    MechanismDefinition,
)

EXAMPLES_DIR = (
    Path(__file__).parent.parent / "examples"
)

MECHANISM = (
    EXAMPLES_DIR
    / "pivot_shift_mechanism.csv"
).read_text(encoding="utf-8")
UNBRAKED = (
    EXAMPLES_DIR
    / "target_curve_unbraked.csv"
).read_text(encoding="utf-8")
BRAKED = (
    EXAMPLES_DIR
    / "target_curve_braked.csv"
).read_text(encoding="utf-8")


def test_load_inputs_from_example_files():
    inputs = load_pivot_shift_inputs(
        MECHANISM,
        UNBRAKED,
        BRAKED,
    )
    assert inputs.shift_lever_id in {1, 3}
    pair = pivot_pair_candidates(
        inputs.definition,
    )
    assert pair == [{1, 3}]
    assert inputs.input_angles
    assert len(inputs.input_angles) == len(
        inputs.unbraked.input_angles,
    )


def test_inputs_from_parts_rejects_pivot_on():
    mixed = CsvReader.read_mechanism(
        EXAMPLES_DIR
        / "mixed_brake_mechanism.csv",
    )
    unbraked = TargetCurve.from_csv_strict(
        EXAMPLES_DIR
        / "target_curve_unbraked.csv",
    )
    braked = TargetCurve.from_csv_strict(
        EXAMPLES_DIR
        / "target_curve_braked.csv",
    )
    with pytest.raises(WorkflowError):
        pivot_shift_inputs_from_parts(
            mixed,
            unbraked,
            braked,
        )


def test_inputs_from_parts_rejects_no_pair():
    definition = MechanismDefinition(
        tuple(
            lever
            for lever in CsvReader
            .read_mechanism(
                EXAMPLES_DIR
                / "pivot_shift_mechanism.csv",
            )
            .levers
            if lever.id != 3
        ),
    )
    unbraked = TargetCurve.from_csv_strict(
        EXAMPLES_DIR
        / "target_curve_unbraked.csv",
    )
    braked = TargetCurve.from_csv_strict(
        EXAMPLES_DIR
        / "target_curve_braked.csv",
    )
    with pytest.raises(WorkflowError):
        pivot_shift_inputs_from_parts(
            definition,
            unbraked,
            braked,
        )


def test_parameter_template():
    inputs = load_pivot_shift_inputs(
        MECHANISM,
        UNBRAKED,
        BRAKED,
    )
    template = (
        pivot_shift_parameter_template(inputs)
    )
    names = {
        parameter.name
        for parameter in template.parameters
    }
    assert (
        f"lever.{inputs.shift_lever_id}"
        ".shift.x"
        in names
    )
    assert (
        f"lever.{inputs.shift_lever_id}"
        ".shift.y"
        in names
    )
    # The pair partner owns no pivot parameters.
    partner = next(
        lever_id
        for lever_id in (1, 3)
        if lever_id != inputs.shift_lever_id
    )
    for axis in ("x", "y", "z"):
        assert (
            f"lever.{partner}.pivot.{axis}"
            not in names
        )


def test_short_run_produces_result():
    inputs = load_pivot_shift_inputs(
        MECHANISM,
        UNBRAKED,
        BRAKED,
    )
    template = (
        pivot_shift_parameter_template(inputs)
    )
    settings = PivotShiftSettings(
        population_size=10,
        children_count=10,
        selection_count=4,
        max_generations=3,
        stagnation_limit=5,
        workers=1,
        seed=42,
    )
    progressed = []
    bests = []

    def progress(generation, best_score):
        progressed.append(
            (generation, best_score),
        )

    def on_best(generation, values):
        bests.append((generation, dict(values)))

    result = run_pivot_shift_optimization(
        inputs,
        template,
        settings,
        progress=progress,
        on_best=on_best,
    )
    assert result.best_score < float("inf")
    assert result.generations >= 1
    assert len(result.fitness_history) >= 2
    assert result.generations == len(
        progressed,
    )
    assert result.shift_mm is not None
    assert (
        result.optimized_definition
        is not None
    )
    assert result.mechanism_csv.startswith("id,")
    assert (
        len(result.unbraked_target_deg)
        == len(result.input_angles_deg)
    )
    assert (
        len(result.unbraked_actual_deg)
        == len(result.input_angles_deg)
    )
    assert result.braked_buildable is True
    assert (
        len(result.braked_actual_deg)
        == len(result.input_angles_deg)
    )
    if bests:
        _generation, values = bests[-1]
        assert (
            f"lever.{inputs.shift_lever_id}"
            ".shift.x"
            in values
        )


def test_stop_reason_on_stopped_run():
    inputs = load_pivot_shift_inputs(
        MECHANISM,
        UNBRAKED,
        BRAKED,
    )
    template = (
        pivot_shift_parameter_template(inputs)
    )
    settings = PivotShiftSettings(
        population_size=6,
        children_count=6,
        selection_count=2,
        max_generations=50,
        stagnation_limit=50,
        workers=1,
        seed=7,
    )

    from gui.workflow import (
        OptimizationStopped,
    )

    def progress(generation, best_score):
        if generation >= 2:
            raise OptimizationStopped(
                "Test-Stopp",
            )

    result = run_pivot_shift_optimization(
        inputs,
        template,
        settings,
        progress=progress,
    )
    assert (
        result.stop_reason
        == "stopped_by_user"
    )
    assert result.generations >= 1


def test_minimum_transmission_angle_reported():
    inputs = load_pivot_shift_inputs(
        MECHANISM,
        UNBRAKED,
        BRAKED,
    )
    template = (
        pivot_shift_parameter_template(inputs)
    )
    settings = PivotShiftSettings(
        population_size=8,
        children_count=8,
        selection_count=3,
        max_generations=2,
        stagnation_limit=5,
        workers=1,
        seed=42,
    )
    result = run_pivot_shift_optimization(
        inputs,
        template,
        settings,
    )
    assert 0.0 < (
        result.min_transmission_angle_deg
        <= 90.0 + 1e-9
    )
    assert math.isfinite(
        result.best_score,
    )
