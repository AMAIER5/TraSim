import math
import random

import pytest

from analysis.curve_fitness import CurveFitness
from analysis.target_curve import TargetCurve

from mechanism_io.csv_reader import CsvReader

from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)

from optimization.evolution_engine import (
    EvolutionEngine,
)
from optimization.parameter import (
    Parameter,
)
from optimization.parameter_mutation import (
    ParameterMutation,
)
from optimization.parameter_set import (
    ParameterSet,
)
from optimization.population_factory import (
    PopulationFactory,
)
from optimization.reproduction import (
    Reproduction,
)
from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)

from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import (
    MotionRange,
)

from simulation.simulation_result import (
    SimulationResult,
)

def create_parameter_template():

    return ParameterSet(
        (
            Parameter(
                name="lever.1.length",
                minimum=20,
                maximum=40,
                value=30,
            ),
            Parameter(
                name="lever.2.length",
                minimum=80,
                maximum=120,
                value=100,
            ),
        )
    )

def main():

    # deine Test-CSV
    definition = CsvReader.read_mechanism(
        "tests/data/simple_stage.csv"
    )

    builder = CsvMechanismBuilder(
        definition
    )

    # einen Parametersatz erzeugen
    rng = random.Random(42)
    population = PopulationFactory(
        random_generator=rng,
    ).create(
        create_parameter_template(),
        size=1,
    )

    candidate = population[0]
    mechanism = builder.build(
        candidate
    )

    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=math.radians(-40),
            max_angle=math.radians(80),
            step=math.radians(5),
        )
    )

    results = simulator.simulate(
        mechanism
    )

    result = results[0]


    print()
    print("Simulation result")
    print("=================")
    print("success:", result.success)


if __name__ == "__main__":
    main()