import math
import random

from mechanism_io.csv_reader import CsvReader
from mechanics.csv_mechanism_builder import CsvMechanismBuilder

from optimization.parameter import Parameter
from optimization.parameter_set import ParameterSet
from optimization.population_factory import PopulationFactory

from simulation.mechanism_simulator import MechanismSimulator
from simulation.motion_range import MotionRange

from solver.objective import stage_error


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


def test_debug_simulation(
    simple_stage_csv,
):

    print("Loading mechanism...")

    definition = CsvReader.read_mechanism(
        simple_stage_csv,
    )

    print(definition)

    builder = CsvMechanismBuilder(
        definition,
    )

    rng = random.Random(42)

    population = PopulationFactory(
        random_generator=rng,
    ).create(
        create_parameter_template(),
        size=1,
    )

    mechanism = builder.build(
        population[0],
    )

    stage = mechanism.stages[0]

    print()
    print("Built mechanism")
    print("================")
    print(stage)

    print()
    print("Residual around reference")
    print("=========================")

    for angle in range(-180, 181, 30):

        value = stage_error(
            stage,
            math.radians(-30),
            math.radians(angle),
        )

        print(
            f"{angle:6.0f}° : {value: .6f}"
        )

    print()
    print("Starting simulation...")

    simulator = MechanismSimulator(
        motion=MotionRange(
            start_angle=math.radians(-30),
            max_angle=math.radians(60),
            step=math.radians(5),
        )
    )

    results = simulator.simulate(
        mechanism,
    )

    print()
    print("Returned")
    print("========")

    print(type(results))
    print("count:", len(results))

    for index, result in enumerate(results):

        print()
        print(f"Simulation {index}")
        print("================")

        print("success:", result.success)
        print("samples:", len(result.input_angles))
        print(
            "blocked_at:",
            None
            if result.blocked_at is None
            else math.degrees(result.blocked_at),
        )

        print()
        print("Curve")
        print("-----")

        previous = None

        for input_angle, output_angle in zip(
            result.input_angles,
            result.output_angles,
        ):

            output_deg = math.degrees(output_angle)

            if previous is None:

                print(
                    f"{math.degrees(input_angle):7.2f}°"
                    f" -> "
                    f"{output_deg:8.2f}°"
                )

            else:

                delta = abs(
                    output_deg - previous
                )

                print(
                    f"{math.degrees(input_angle):7.2f}°"
                    f" -> "
                    f"{output_deg:8.2f}°"
                    f"   delta={delta:7.2f}°"
                )

            previous = output_deg