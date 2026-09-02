"""
tests/test_curve_fitness.py
"""

from __future__ import annotations

from analysis.curve_fitness import (
    CurveFitness,
)
from analysis.target_curve import (
    TargetCurve,
)
from analysis.transfer_curve import (
    TransferCurve,
)
from simulation.simulation_result import SimulationResult

def create_transfer_curve() -> TransferCurve:

    return TransferCurve(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            0.0,
            1.0,
            2.0,
        ),
    )


def test_identical_curve_has_zero_error():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    result = fitness(
        create_transfer_curve(),
    )

    assert result == 0.0


def test_shifted_curve_has_positive_error():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    simulated = TransferCurve(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            1.0,
            2.0,
            3.0,
        ),
    )

    assert fitness(simulated) > 0.0


def test_target_is_sampled_at_input_angles():

    sampled: dict[str, tuple[float, ...]] = {}

    class RecordingTargetCurve(TargetCurve):

        def sample(
            self,
            input_angles: tuple[float, ...],
        ) -> TransferCurve:

            sampled["angles"] = input_angles

            return super().sample(
                input_angles,
            )

    transfer = create_transfer_curve()

    fitness = CurveFitness(
        target_curve=RecordingTargetCurve(
            function=lambda angle: angle,
        ),
    )

    fitness(transfer)

    assert sampled["angles"] == transfer.input_angles


def test_metric_is_cached():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    curve = create_transfer_curve()

    fitness(curve)

    assert len(fitness._cache) == 1

    fitness(curve)

    assert len(fitness._cache) == 1

def test_evaluate_uses_last_stage_result():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    stage1 = SimulationResult(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            10.0,
            11.0,
            12.0,
        ),
        success=True,
    )

    stage2 = SimulationResult(
        input_angles=(
            0.0,
            1.0,
            2.0,
        ),
        output_angles=(
            0.0,
            1.0,
            2.0,
        ),
        success=True,
    )

    result = fitness.evaluate(
        (
            stage1,
            stage2,
        )
    )

    assert result == 0.0
    
def test_uses_last_stage_output():

    fitness = CurveFitness(
        target_curve=TargetCurve(
            function=lambda angle: angle,
        ),
    )

    stage1 = SimulationResult(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            10.0,
            11.0,
        ),
        success=True,
    )

    stage2 = SimulationResult(
        input_angles=(
            0.0,
            1.0,
        ),
        output_angles=(
            0.0,
            1.0,
        ),
        success=True,
    )

    result = fitness.evaluate(
        (
            stage1,
            stage2,
        )
    )

    assert result == 0.0