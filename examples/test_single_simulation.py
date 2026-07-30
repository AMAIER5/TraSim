"""
examples/test_single_simulation.py

Test simulation of one mechanism built from
a fixed parameter set.
"""

from __future__ import annotations

import math

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
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


# -------------------------------------------------
# Test parameter set
# -------------------------------------------------

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


# -------------------------------------------------
# Build mechanism
# -------------------------------------------------

builder = StandardMechanismBuilder()

mechanism = builder.build(
    parameters
)

stage = mechanism.stages[0]


print("Stage geometry")
print("================")

print(
    f"Rod length: {stage.rod_length:.6f}"
)

print(
    f"Input offset: "
    f"{math.degrees(stage.input_angle_offset):.3f} deg"
)

print(
    f"Output offset: "
    f"{math.degrees(stage.output_angle_offset):.3f} deg"
)


# -------------------------------------------------
# Simulation
# -------------------------------------------------

motion = MotionRange(
    start_angle=math.radians(-50),
    max_angle=math.radians(100),
    step=math.radians(10),
    direction=1,
)


simulator = MechanismSimulator(
    motion=motion,
)


results = simulator.simulate(
    mechanism,
)


# -------------------------------------------------
# Result
# -------------------------------------------------

print()
print("Simulation")
print("================")

for index, stage_result in enumerate(results):

    print(
        f"Stage {index}:"
    )

    print(
        "Success:",
        stage_result.success,
    )

    print(
        "Input angles:"
    )

    print(
        [
            round(
                math.degrees(angle),
                2,
            )
            for angle
            in stage_result.input_angles
        ]
    )

    print(
        "Output angles:"
    )

    print(
        [
            round(
                math.degrees(angle),
                2,
            )
            for angle
            in stage_result.output_angles
        ]
    )

    if not stage_result.success:
        print(
            "Blocked at:",
            stage_result.blocked_at,
        )