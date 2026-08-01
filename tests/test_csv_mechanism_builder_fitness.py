"""
tests/test_csv_mechanism_builder_fitness.py

End-to-end test:
CSV -> Definition -> ParameterSet -> Simulation -> Fitness
"""

from __future__ import annotations
from math import radians

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)

from mechanism_io import CsvReader

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
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


class OutputRangeFitness:
    def evaluate(
        self,
        simulation,
    ) -> float:

        result = simulation[0]

        print("success:", result.success)
        print("blocked_at:", result.blocked_at)
        print("input:", result.input_angles)
        print("output:", result.output_angles)

        if not result.success:
            return -1

        return (
            result.output_angles[-1]
            -
            result.output_angles[0]
        )


def create_parameter_set(
    length: float,
) -> ParameterSet:

    return ParameterSet(
        parameters=(
            Parameter(
                name="lever.1.length",
                minimum=40,
                maximum=100,
                value=length,
            ),
        )
    )


def test_csv_parameter_changes_fitness(
    example_mechanism_csv,
):

    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    optimizer = MechanismOptimizer(
        builder=CsvMechanismBuilder(
            definition,
        ),
        simulator=MechanismSimulator(
            motion=MotionRange(
                start_angle=0,
                max_angle=radians(10),
                step=radians(2),
            ),
        ),
        fitness=OutputRangeFitness(),
    )

    fitness_a = optimizer.evaluate(
        create_parameter_set(
            50,
        )
    )

    fitness_b = optimizer.evaluate(
        create_parameter_set(
            90,
        )
    )

    assert fitness_a != fitness_b