"""
tests/test_optimization_pipeline.py

End-to-end test for optimization pipeline.
"""

from __future__ import annotations

from mechanics.mechanism_factory import (
    MechanismFactory,
)

from mechanics.standard_mechanism_builder import (
    StandardMechanismBuilder,
)

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)

from optimization.parameter import (
    Parameter,
)

from optimization.parameter_set import (
    ParameterSet,
)


def create_parameters():

    return ParameterSet(
        (
            Parameter(
                name="input_lever_length",
                minimum=10.0,
                maximum=100.0,
                value=40.0,
            ),
            Parameter(
                name="output_lever_length",
                minimum=10.0,
                maximum=100.0,
                value=30.0,
            ),
            Parameter(
                name="rod_length",
                minimum=20.0,
                maximum=200.0,
                value=120.0,
            ),
        )
    )


def test_complete_optimization_pipeline():

    class DummySimulator:

        def simulate(
            self,
            mechanism,
        ):
            return {
                "stages": len(mechanism.stages),
            }


    class DummyFitness:

        def evaluate(
            self,
            result,
        ) -> float:
            return float(
                result["stages"]
            )
        
    builder = (
        StandardMechanismBuilder()
    )

    factory = MechanismFactory(
        builder=builder.build,
    )
    optimizer = MechanismOptimizer(
        builder=builder,
        simulator=DummySimulator(),
        fitness=DummyFitness(),
    )

    score = optimizer.evaluate(
        create_parameters()
    )

    assert score == 1.0