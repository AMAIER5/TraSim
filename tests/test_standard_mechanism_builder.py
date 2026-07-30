"""
tests/test_standard_mechanism_builder.py
"""

import math

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)

from optimization.parameter import Parameter
from optimization.parameter_set import ParameterSet

from simulation.motion_range import MotionRange
from simulation.mechanism_simulator import (
    MechanismSimulator,
)


def test_standard_mechanism_simulation():

    parameters = ParameterSet(
        parameters=(
            Parameter(
                name="input_lever_length",
                minimum=50.0,
                maximum=100.0,
                value=70.0,
            ),
            Parameter(
                name="output_lever_length",
                minimum=50.0,
                maximum=100.0,
                value=70.0,
            ),
            Parameter(
                name="input_angle_offset",
                minimum=-math.pi,
                maximum=math.pi,
                value=math.radians(-20),
            ),
            Parameter(
                name="output_angle_offset",
                minimum=-math.pi,
                maximum=math.pi,
                value=math.radians(-25),
            ),
        )
    )

    builder = StandardMechanismBuilder()

    mechanism = builder.build(
        parameters
    )

    stage = mechanism.stages[0]

    print()
    print("Reference")
    print("=========")

    print(
        "rod:",
        stage.rod_length
    )

    print(
        "input ref:",
        stage.input_endpoint
    )

    print(
        "output ref:",
        stage.output_endpoint
    )

    print()
    print("At -50 deg")
    print("==========")

    input_point = stage.input_position(
        math.radians(-50)
    )

    print(
        "input:",
        input_point
    )

    print(
        "distance to output ref:",
        (
            input_point
            -
            stage.output_endpoint
        ).norm()
    )

    motion = MotionRange(
        start_angle=0.0,
        max_angle=math.radians(100),
        step=math.radians(10),
        direction=1,
    )

    simulator = MechanismSimulator(
        motion=motion,
    )

    results = simulator.simulate(
        mechanism
    )

    assert len(results) == 1

    result = results[0]

    assert result.success

    assert len(result.input_angles) > 0
    assert len(result.output_angles) > 0

    print()
    print("Simulation")
    print("==========")

    for inp, out in zip(
        result.input_angles,
        result.output_angles,
    ):
        print(
            f"{math.degrees(inp):8.2f} -> "
            f"{math.degrees(out):8.2f}"
        )