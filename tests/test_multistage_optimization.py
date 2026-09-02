"""
tests/test_multistage_optimization.py

Tests for optimization of complete multi-stage mechanisms.
"""

from __future__ import annotations

import math

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)

from mechanism_io.csv_reader import (
    CsvReader,
)

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
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


def test_optimizer_accepts_multistage_mechanism(
    example_mechanism_csv,
):
    """
    MechanismOptimizer evaluates complete mechanisms
    instead of single stages.
    """

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    builder = CsvMechanismBuilder(
        definition,
    )

    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=0.0,
            max_angle=math.radians(20),
            step=math.radians(10),
        ),
    )

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    optimizer = MechanismOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
    )

    score = optimizer.evaluate(
        ParameterSet(()),
    )

    assert score >= 0.0


def test_optimizer_uses_complete_output_chain(
    example_mechanism_csv,
):
    """
    The optimization pipeline must evaluate
    the final stage output, not the first stage.
    """

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    builder = CsvMechanismBuilder(
        definition,
    )

    mechanism = builder.build(
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

    assert len(results) > 1

    final_stage = results[-1]

    assert final_stage.success
    assert final_stage.output_angles != results[0].output_angles