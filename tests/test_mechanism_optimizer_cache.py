"""
tests/test_mechanism_optimizer_cache.py

Tests for MechanismOptimizer cache behavior.
"""

from __future__ import annotations

import math

from mechanics.mechanism import Mechanism

from optimization.mechanism_optimizer import (
    MechanismOptimizer,
)
from optimization.parameter import (
    Parameter,
)
from optimization.parameter_set import (
    ParameterSet,
)

from simulation.motion_range import (
    MotionRange,
)

from solver.solver_precision import (
    SolverPrecision,
)


class CountingBuilder:
    def __init__(self):

        self.calls = 0


    def build(
        self,
        parameters: ParameterSet,
    ) -> Mechanism:

        self.calls += 1

        return Mechanism(
            stages=(),
        )



class CountingSimulator:

    def __init__(
        self,
        *,
        motion: MotionRange,
        precision: SolverPrecision | None = None,
    ) -> None:

        self.calls = 0

        self._motion = motion

        self._precision = precision


    @property
    def motion(
        self,
    ) -> MotionRange:

        return self._motion


    @property
    def precision(
        self,
    ) -> SolverPrecision | None:

        return self._precision


    @property
    def stage_limit(
        self,
    ) -> int | None:

        return None


    def simulate(
        self,
        mechanism: Mechanism,
    ):

        self.calls += 1

        return "simulation"



class CountingFitness:

    def __init__(self):

        self.calls = 0


    def evaluate(
        self,
        simulation,
    ) -> float:

        self.calls += 1

        return 1.0



def create_parameters() -> ParameterSet:

    return ParameterSet(
        (
            Parameter(
                name="length",
                minimum=10,
                maximum=100,
                value=50,
            ),
        )
    )



def create_motion(
    max_angle: float,
) -> MotionRange:

    return MotionRange(
        start_angle=0,
        max_angle=math.radians(
            max_angle,
        ),
        step=math.radians(
            1,
        ),
    )



def create_optimizer(
    simulator,
):

    builder = CountingBuilder()

    fitness = CountingFitness()

    optimizer = MechanismOptimizer(
        builder=builder,
        simulator=simulator,
        fitness=fitness,
    )

    return (
        optimizer,
        builder,
        fitness,
    )



def test_same_parameter_same_simulation_is_cache_hit():

    simulator = CountingSimulator(
        motion=create_motion(10),
    )

    optimizer, _, _ = create_optimizer(
        simulator,
    )

    parameters = create_parameters()

    first = optimizer.evaluate(
        parameters,
    )

    second = optimizer.evaluate(
        parameters,
    )

    assert first == second
    assert simulator.calls == 1



def test_same_parameter_different_precision_is_cache_miss():

    parameters = create_parameters()

    simulator_a = CountingSimulator(
        motion=create_motion(10),
        precision=SolverPrecision(
            tolerance=1e-6,
        ),
    )

    simulator_b = CountingSimulator(
        motion=create_motion(10),
        precision=SolverPrecision(
            tolerance=1e-10,
        ),
    )

    optimizer_a, _, _ = create_optimizer(
        simulator_a,
    )

    optimizer_b, _, _ = create_optimizer(
        simulator_b,
    )

    optimizer_a.evaluate(
        parameters,
    )

    optimizer_b.evaluate(
        parameters,
    )

    assert simulator_a.calls == 1
    assert simulator_b.calls == 1



def test_same_parameter_different_motion_is_cache_miss():

    parameters = create_parameters()

    simulator_a = CountingSimulator(
        motion=create_motion(10),
    )

    simulator_b = CountingSimulator(
        motion=create_motion(20),
    )

    optimizer_a, _, _ = create_optimizer(
        simulator_a,
    )

    optimizer_b, _, _ = create_optimizer(
        simulator_b,
    )

    optimizer_a.evaluate(
        parameters,
    )

    optimizer_b.evaluate(
        parameters,
    )

    assert simulator_a.calls == 1
    assert simulator_b.calls == 1



def test_same_parameter_same_configuration_uses_cache():

    simulator = CountingSimulator(
        motion=create_motion(10),
        precision=SolverPrecision(
            tolerance=1e-8,
        ),
    )

    optimizer, builder, fitness = create_optimizer(
        simulator,
    )

    parameters = create_parameters()

    optimizer.evaluate(
        parameters,
    )

    optimizer.evaluate(
        parameters,
    )

    optimizer.evaluate(
        parameters,
    )

    assert builder.calls == 1
    assert simulator.calls == 1
    assert fitness.calls == 1