"""
tests/test_stage_continuity.py

Verify continuous motion of a CSV-defined mechanism.

The test validates:
- successful simulation
- continuous output motion
- bounded velocity changes
"""

from __future__ import annotations

import math

from mechanism_io.csv_reader import CsvReader
from mechanics.csv_mechanism_builder import CsvMechanismBuilder
from simulation.mechanism_simulator import MechanismSimulator
from simulation.motion_range import MotionRange


def create_parameter_template():
    from optimization.parameter import Parameter
    from optimization.parameter_set import ParameterSet

    return ParameterSet(
        (
            Parameter(
                name="lever.1.length",
                minimum=30,
                maximum=50,
                value=40,
            ),
            Parameter(
                name="lever.2.length",
                minimum=90,
                maximum=110,
                value=100,
            ),
        )
    )


def test_simple_stage_has_continuous_motion(
    simple_stage_csv,
):
    """
    Verify that a CSV-defined mechanism follows one
    continuous physical motion branch.
    """

    definition = CsvReader.read_mechanism(
        simple_stage_csv,
    )

    mechanism = CsvMechanismBuilder(
        definition,
    ).build(
        create_parameter_template(),
    )


    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=math.radians(-30),
            max_angle=math.radians(10),
            step=math.radians(5),
        ),
    )


    results = simulator.simulate(
        mechanism,
    )


    assert len(results) == 1


    result = results[0]


    assert result.success, (
        "Simulation blocked at "
        f"{math.degrees(result.blocked_at):.2f}°"
        if result.blocked_at is not None
        else "Unknown position"
    )


    assert len(result.input_angles) > 2


    #
    # Analyze output continuity
    #

    output_deltas: list[float] = []


    for previous, current in zip(
        result.output_angles,
        result.output_angles[1:],
    ):

        output_deltas.append(
            current - previous
        )


    #
    # First derivative:
    # output change per input step
    #

    max_output_jump = max(
        abs(delta)
        for delta in output_deltas
    )


    assert max_output_jump < math.radians(20), (
        "Detected output branch jump: "
        f"{math.degrees(max_output_jump):.2f}°"
    )



    #
    # Second derivative:
    # change of output velocity
    #

    velocity_changes: list[float] = []


    for previous, current in zip(
        output_deltas,
        output_deltas[1:],
    ):

        velocity_changes.append(
            current - previous
        )


    max_velocity_change = max(
        abs(change)
        for change in velocity_changes
    )

    assert result.success

    assert len(result.input_angles) > 2

    assert len(result.input_angles) == len(
        result.output_angles
    )

    assert max_velocity_change < math.radians(15), (
        "Detected velocity discontinuity: "
        f"{math.degrees(max_velocity_change):.2f}°"
    )