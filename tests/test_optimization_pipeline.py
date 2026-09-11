"""
tests/test_optimization_pipeline.py

End-to-end test for optimization pipeline.
"""

from __future__ import annotations

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
            Parameter(
                name="input_angle_offset",
                minimum=-3.141592653589793,
                maximum=3.141592653589793,
                value=0.0,
            ),
            Parameter(
                name="output_angle_offset",
                minimum=-3.141592653589793,
                maximum=3.141592653589793,
                value=0.0,
            ),
        )
    )


def test_complete_optimization_pipeline():

    class DummySimulator:

        @property
        def motion(self):
            from simulation.motion_range import MotionRange
            return MotionRange(
                start_angle=0.0,
                max_angle=1.0,
                step=0.5,
            )

        @property
        def precision(self):
            return None

        @property
        def stage_limit(self):
            return None

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
            validation=None,
        ) -> float:
            return float(
                result["stages"]
            )
        
    builder = (
        StandardMechanismBuilder()
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