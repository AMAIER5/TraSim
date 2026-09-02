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
    return MechanismSimulator(
        motion=MotionRange(
            start_angle=0.0,
            max_angle=math.radians(5),
            step=math.radians(1),
        ),
    )


def get_final_output(
    simulator,
    builder,
    parameters,
):
    mechanism = builder.build(parameters)

    results = simulator.simulate(mechanism)

    print()
    print("SIMULATION")
    print("==========")

    for index, result in enumerate(results, start=1):
        print(
            f"Stage {index}: "
            f"success={result.success}, "
            f"blocked={result.blocked_at}"
        )

        if result.output_angles:
            print(
                " last output:",
                result.output_angles[-1],
            )

    assert len(results) > 1

    assert all(
        result.success
        for result in results
    )

    return results[-1].output_angles


def test_stage1_parameter_changes_final_output(
    example_mechanism_csv,
):
    """
    A change at the first stage must influence
    the final mechanism output.
    """

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    builder = CsvMechanismBuilder(
        definition,
    )

    simulator = create_simulator()

    default = ParameterSet(
        parameters=(),
    )

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
    
    output_default = get_final_output(
        simulator,
        builder,
        default,
    )

    output_changed = get_final_output(
        simulator,
        builder,
        changed,
    )
    
    assert output_default != output_changed


def test_last_stage_output_lever_changes_final_output(
    example_mechanism_csv,
):
    """
    A change at the last stage must influence
    the final mechanism output.
    """

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    builder = CsvMechanismBuilder(
        definition,
    )

    simulator = create_simulator()

    default = ParameterSet(
        parameters=(),
    )

    changed = ParameterSet(
        parameters=(
            Parameter(
                name="lever.4.length",
                minimum=20,
                maximum=80,
                value=70,
            ),
        ),
    )
    
    print("CHANGED PARAMETERS:")
    print(changed.values())
    
    mechanism_default = builder.build(default)
    mechanism_changed = builder.build(changed)
    
    for name, mechanism in (
        ("DEFAULT", mechanism_default),
        ("CHANGED", mechanism_changed),
    ):
        print()
        print(name)

        for index, stage in enumerate(
            mechanism.stages,
            start=1,
        ):
            print(
                index,
                math.degrees(stage.output_angle_min),
                math.degrees(stage.output_angle_max),
            )

    print(
        mechanism_default.stages[-1].output_lever.length
    )

    print(
        mechanism_changed.stages[-1].output_lever.length
    )
    
    output_default = get_final_output(
        simulator,
        builder,
        default,
    )

    output_changed = get_final_output(
        simulator,
        builder,
        changed,
    )

    assert output_default != output_changed