"""
Example:
Optimize a single mechanism stage.

This represents the intended user workflow.
"""

from __future__ import annotations

import random

from mechanics.mechanism_factory import (
    MechanismFactory,
)

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)

from optimization.evolution_engine import (
    EvolutionEngine,
)

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)

from optimization.optimizer_runner import (
    OptimizerRunner,
)

from optimization.parameter import (
    Parameter,
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

from optimization.parameter_mutation import (
    ParameterMutation,
)

from optimization.selection import (
    Selection,
)


def create_parameter_template():

    return ParameterSet(
        (
            Parameter(
                name="input_lever_length",
                minimum=20.0,
                maximum=80.0,
                value=40.0,
            ),

            Parameter(
                name="output_lever_length",
                minimum=20.0,
                maximum=80.0,
                value=30.0,
            ),

            Parameter(
                name="rod_length",
                minimum=80.0,
                maximum=160.0,
                value=120.0,
            ),
        )
    )


def main():

    #
    # 1. Create initial population
    #

    population = (
        PopulationFactory(
            random_generator=random.Random(1)
        )
        .create(
            create_parameter_template(),
            size=20,
        )
    )


    #
    # 2. Connect mechanism generation
    #

    builder = (
        StandardMechanismBuilder()
    )

    factory = MechanismFactory(
        builder=builder.build,
    )


    #
    # 3. Define evaluation
    #

    mechanism_optimizer = (
        MechanismOptimizer(
            mechanism_factory=factory.create,

            # Placeholder simulation
            simulator=lambda mechanism:
                len(
                    mechanism.stages
                ),

            # Placeholder fitness
            fitness=lambda result:
                float(result),
        )
    )


    #
    # 4. Create evolution engine
    #

    engine = EvolutionEngine(
        population=population,

        evaluator=(
            mechanism_optimizer.evaluate
        ),

        selection=Selection(),

        reproduction=Reproduction(
            mutation=ParameterMutation(
                random_generator=random.Random(2)
            )
        ),
    )


    #
    # 5. Run optimization
    #

    runner = OptimizerRunner(
        engine=engine,
    )

    result = runner.run(
        generations=10,
        children_count=10,
    )


    #
    # 6. Display result
    #

    best = result[0]

    print(
        "Best candidate:"
    )

    print(
        best
    )


if __name__ == "__main__":

    main()