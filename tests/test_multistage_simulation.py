from __future__ import annotations

import math

from mechanism_io.csv_reader import CsvReader

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
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
from tests.conftest import simple_multistage_csv


def test_multistage_simulation_returns_all_stages(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    mechanism = CsvMechanismBuilder(
        definition,
    ).build(
        ParameterSet(()),
    )

    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=0.0,
            max_angle=0.0,
            step=math.radians(10),
        ),
    )

    results = simulator.simulate(
        mechanism,
    )

    assert len(results) == 3

    assert all(
        result.success
        for result in results
    )


def test_multistage_results_preserve_stage_order(
    simple_multistage_csv,
):
    """
    Result order follows mechanism stage order.
    """

    definition = CsvReader.read_mechanism(
        simple_multistage_csv,
    )

    mechanism = CsvMechanismBuilder(
        definition,
    ).build(
        ParameterSet(()),
    )

    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=0.0,
            max_angle=math.radians(20),
            step=math.radians(10),
        ),
    )

    results = simulator.simulate(
        mechanism,
    )

    assert len(results) == 2

    stage1, stage2 = results

    assert len(stage1.input_angles) > 0
    assert len(stage2.input_angles) > 0

    assert stage1.success
    assert stage2.success

    assert stage2.input_angles == stage1.output_angles


def test_stage_chain_changes_with_first_stage_output(
    example_mechanism_csv,
):
    """
    A change in the first stage propagates
    through the complete kinematic chain.
    """

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    mechanism = CsvMechanismBuilder(
        definition,
    ).build(
        ParameterSet(()),
    )

    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=0.0,
            max_angle=math.radians(30),
            step=math.radians(10),
        ),
    )

    results = simulator.simulate(
        mechanism,
    )

    stage1 = results[0]
    stage3 = results[-1]

    assert stage1.output_angles != ()

    assert stage3.output_angles != ()

    assert stage3.output_angles != (
        stage1.output_angles
    )