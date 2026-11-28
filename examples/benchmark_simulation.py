"""
examples/benchmark_simulation.py

Benchmark for the simulation hot path.

Builds a fixed three-stage planar mechanism and simulates
481 input angles (0.5 deg step) through all three stages,
then prints the wall-clock time of the repeated runs.

Run directly::

    python examples/benchmark_simulation.py

or::

    python -m examples.benchmark_simulation

The script also reports whether the optional Cython-transpiled
native residual module is active (see docs/performance.md)::

    pip install cython
    python setup.py build_ext --inplace
"""

from __future__ import annotations

import math
import time

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.mechanism import Mechanism
from mechanics.stage import Stage
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.motion_range import MotionRange


def build_mechanism() -> Mechanism:
    """
    Fixed three-stage planar chain on the X axis.
    """

    axis = Vector3D(0.0, 0.0, 1.0)
    reference = Vector3D(1.0, 0.0, 0.0)

    def lever(
        x: float,
        length: float,
    ) -> Lever:

        return Lever(
            pivot=Point3D(x, 0.0, 0.0),
            axis=axis,
            length=length,
            reference_direction=reference,
        )

    input_lever_1 = lever(0.0, 70.0)
    output_lever_1 = lever(120.0, 60.0)
    output_lever_2 = lever(220.0, 65.0)
    output_lever_3 = lever(300.0, 55.0)

    stage_1 = Stage.from_reference_position(
        input_lever_1,
        output_lever_1,
        math.radians(-20.0),
        math.radians(-25.0),
    )

    stage_2 = Stage.from_reference_position(
        output_lever_1,
        output_lever_2,
        math.radians(-25.0),
        math.radians(100.0),
    )

    stage_3 = Stage.from_reference_position(
        output_lever_2,
        output_lever_3,
        math.radians(100.0),
        math.radians(60.0),
    )

    return Mechanism(
        stages=(
            stage_1,
            stage_2,
            stage_3,
        ),
    )


def run_once() -> tuple[float, int]:
    """
    Run one full simulation and return (seconds, stage points).
    """

    mechanism = build_mechanism()

    motion = MotionRange(
        start_angle=math.radians(-120.0),
        max_angle=math.radians(240.0),
        step=math.radians(0.5),
    )

    simulator = MechanismSimulator(
        motion=motion,
    )

    start = time.perf_counter()

    results = simulator.simulate(mechanism)

    elapsed = time.perf_counter() - start

    if not all(result.success for result in results):
        raise RuntimeError("Benchmark mechanism blocked.")

    points = sum(
        len(result.output_angles)
        for result in results
    )

    return elapsed, points


def main() -> None:

    import solver._rod_residual as rod_residual

    native = rod_residual.__file__.endswith(".so")

    print(
        f"residual module: "
        f"{'native (Cython)' if native else 'pure Python'}"
    )

    repetitions = 5

    times = []

    for _ in range(repetitions):
        elapsed, points = run_once()
        times.append(elapsed)

    best = min(times)

    print(
        f"stage points : {points} "
        f"(3 stages x {points // 3} angles)"
    )

    print(
        f"best of {repetitions}: {best:.4f} s "
        f"({points / best:.0f} stage points/s)"
    )


if __name__ == "__main__":

    main()
