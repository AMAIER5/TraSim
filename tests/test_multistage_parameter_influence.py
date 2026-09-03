"""
tests/test_multistage_parameter_influence.py

Verify that parameter changes propagate through
the complete multi-stage mechanism.
"""

from __future__ import annotations

import math

from mechanism_io.csv_reader import CsvReader

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_set import (
    ParameterSet,
)

from simulation.mechanism_simulator import (
    MechanismSimulator,
)

from simulation.motion_range import (
    MotionRange,
)

def create_simulator() -> MechanismSimulator:
    """Use a conservative motion range for 2-stage mechanism."""
    return MechanismSimulator(
        motion=MotionRange(
            start_angle=0.0,
            max_angle=math.radians(20.0),  # 20° range
            step=math.radians(5.0),
        ),
    )

def get_final_output(
    simulator,
    builder,
    parameters,
):
    mechanism = builder.build(parameters)
    results = simulator.simulate(mechanism)

    assert len(results) > 1, f"Expected {len(results)} stages"
    assert all(r.success for r in results), (
        f"Stages blocked: {[not r.success for r in results]}"
    )
    return results[-1].output_angles

def test_stage1_parameter_changes_final_output(
    example_mechanism_2stage_csv,  # Use 2-stage fixture
):
    """
    A change at the first stage must influence
    the final mechanism output.
    """
    definition = CsvReader.read_mechanism(
        example_mechanism_2stage_csv
    )

    builder = CsvMechanismBuilder(definition)
    simulator = create_simulator()

    default = ParameterSet(parameters=())

    changed = ParameterSet(
        parameters=(
            Parameter(
                name="lever.1.length",
                minimum=40,
                maximum=100,
                value=70,
            ),
        ),
    )

    output_default = get_final_output(simulator, builder, default)
    output_changed = get_final_output(simulator, builder, changed)

    assert output_default != output_changed

def test_last_stage_output_lever_changes_final_output(
    example_mechanism_2stage_csv,  # Use 2-stage fixture
):
    """
    A change at the last stage must influence
    the final mechanism output.
    """
    definition = CsvReader.read_mechanism(
        example_mechanism_2stage_csv
    )

    builder = CsvMechanismBuilder(definition)
    simulator = create_simulator()

    default = ParameterSet(parameters=())

    changed = ParameterSet(
        parameters=(
            Parameter(
                name="lever.3.length",  # Last stage output lever
                minimum=20,
                maximum=70,
                value=40,
            ),
        ),
    )

    output_default = get_final_output(simulator, builder, default)
    output_changed = get_final_output(simulator, builder, changed)

    assert output_default != output_changed