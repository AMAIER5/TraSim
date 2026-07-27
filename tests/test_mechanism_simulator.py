"""
tests/test_mechanism_simulator.py
"""

from __future__ import annotations

from core.point3d import Point3D
from core.vector3d import Vector3D

from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from mechanics.lever import Lever

from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange
from simulation.simulation_result import (
    SimulationResult,
)


class DummyStageSimulator:

    def __init__(self) -> None:

        self.calls: list[
            tuple[Stage, MotionRange]
        ] = []

    def run(
        self,
        *,
        stage: Stage,
        motion: MotionRange,
    ) -> SimulationResult:

        self.calls.append(
            (
                stage,
                motion,
            )
        )

        return SimulationResult(
            input_angles=(0.0,),
            output_angles=(0.0,),
            success=True,
        )


def create_stage() -> Stage:

    input_lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )

    output_lever = Lever(
        pivot=Point3D(100, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )

    return Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=0.0,
        output_angle=0.0,
    )


def create_motion() -> MotionRange:

    return MotionRange(
        start_angle=0.0,
        max_angle=1.0,
        step=0.5,
    )


def test_simulates_every_stage():

    stage_simulator = DummyStageSimulator()

    motion = create_motion()

    mechanism = Mechanism(
        stages=(
            create_stage(),
            create_stage(),
            create_stage(),
        ),
    )

    result = MechanismSimulator(
        motion=motion,
        stage_simulator=stage_simulator,
    ).simulate(
        mechanism,
    )

    assert len(result) == 3

    assert len(stage_simulator.calls) == 3

    assert all(
        call_motion is motion
        for _, call_motion
        in stage_simulator.calls
    )


def test_preserves_stage_order():

    stage1 = create_stage()
    stage2 = create_stage()

    motion = create_motion()

    stage_simulator = DummyStageSimulator()

    MechanismSimulator(
        motion=motion,
        stage_simulator=stage_simulator,
    ).simulate(
        Mechanism(
            stages=(
                stage1,
                stage2,
            ),
        ),
    )

    assert [
        stage
        for stage, _
        in stage_simulator.calls
    ] == [
        stage1,
        stage2,
    ]