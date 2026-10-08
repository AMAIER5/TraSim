"""Tests for the MixedBrakeBuilder mixer kinematics."""

from __future__ import annotations

import math

import pytest
from mechanism_io.csv_reader import CsvReader
from mechanics.mixed_brake_builder import (
    MechanismBuildError,
    MixedBrakeBuilder,
)
from optimization.parameter_set import ParameterSet
from solver.objective import stage_error

# from mechanics.csv_mechanism_builder import CsvMechanismBuilder


MIXER_CSV = """
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled,pivot_on
1,40,100,60,-40,40,0,0,60,0,0,0,1,,,
2,20,80,50,-60,60,0,50,10,0,0,0,1,1,,lever4@0
3,40,200,100,-30,30,0,180,0,0,0,0,1,2,,
4,20,80,50,-60,60,0,0,10,0,0,0,1,,,
""".strip()


@pytest.fixture
def mixer_definition(tmp_path):
    path = tmp_path / "mixer.csv"
    path.write_text(MIXER_CSV, encoding="utf-8")
    return CsvReader.read_mechanism(path)


def _csv_mechanism(definition):
    from mechanics.csv_mechanism_builder import (
        CsvMechanismBuilder,
    )

    return CsvMechanismBuilder(definition).build(
        ParameterSet(())
    )


def test_beta_zero_matches_legacy_build(mixer_definition):
    builder = MixedBrakeBuilder(mixer_definition)
    mechanism = builder.build(0.0)
    legacy = _csv_mechanism(mixer_definition)
    assert len(mechanism.stages) == len(legacy.stages)
    for stage, reference in zip(
        mechanism.stages,
        legacy.stages,
    ):
        assert stage.rod_length == (
            pytest.approx(reference.rod_length)
        )
        assert stage.input_angle == (
            pytest.approx(reference.input_angle)
        )
        assert stage.output_angle == (
            pytest.approx(reference.output_angle)
        )
        assert stage.input_endpoint == (
            reference.input_endpoint
        )
        assert stage.output_endpoint == (
            reference.output_endpoint
        )


def _all_roots(stage, index: int) -> list[float]:
    """Independent root scan of a stage's output residual
    over its admissible output angle range (grid + bisection,
    no AngleSolver, no Newton)."""
    from solver.objective import stage_error

    def f(angle: float) -> float:
        return stage_error(
            stage,
            stage.input_angle,
            angle,
        )

    lo = stage.output_angle_min
    hi = stage.output_angle_max
    n = 4000
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    vals = [f(x) for x in xs]
    roots: list[float] = []
    for i in range(n):
        if vals[i] == 0.0:
            roots.append(xs[i])
        elif vals[i] * vals[i + 1] < 0.0:
            a, b = xs[i], xs[i + 1]
            fa = vals[i]
            for _ in range(80):
                m = (a + b) / 2.0
                fm = f(m)
                if fa * fm <= 0.0:
                    b = m
                else:
                    a = m
                    fa = fm
            roots.append((a + b) / 2.0)
    if vals[-1] == 0.0:
        roots.append(xs[-1])
    return roots


def test_braked_build_keeps_rod_lengths(mixer_definition):
    builder = MixedBrakeBuilder(mixer_definition)
    reference = builder.build(0.0)
    beta = math.radians(10)
    braked = builder.build(beta)

    # Frozen rod lengths identical to the unbraked build
    assert braked.stages[0].rod_length == (
        pytest.approx(reference.stages[0].rod_length)
    )
    assert braked.stages[1].rod_length == (
        pytest.approx(reference.stages[1].rod_length)
    )

    # The mixer pivot moved with the brake lever endpoint
    unbraked_pivot = (
        reference.stages[0].output_lever.pivot
    )
    braked_pivot = (
        braked.stages[0].output_lever.pivot
    )
    assert braked_pivot != unbraked_pivot

    # Branch-faithful restore: for every stage the restored
    # output angle is the root of the frozen-rod residual that
    # is nearest to the unbraked reference angle (verified
    # with an independent grid + bisection scan).
    for index in (0, 1):
        stage = braked.stages[index]
        roots = _all_roots(stage, index)
        assert roots, "expected at least one root"
        restored = stage.output_angle
        assert any(
            abs(root - restored) < 1.0e-9 for root in roots
        ), "restored angle is not a residual root"
        nearest = min(
            roots,
            key=lambda root: abs(
                root - reference.stages[index].output_angle
            ),
        )
        assert nearest == pytest.approx(restored), (
            f"stage {index}: restored angle is not the "
            "branch nearest the unbraked reference"
        )


def test_braked_output_on_same_branch(mixer_definition):
    """The classic wrong-branch failure jumps from ~0 to a
    far-away solution; the restored angle must stay on the
    branch continuous with the unbraked reference."""
    builder = MixedBrakeBuilder(mixer_definition)
    reference = builder.build(0.0)
    for degrees in (2.5, 5.0, 10.0):
        braked = builder.build(math.radians(degrees))
        for index in (0, 1):
            stage = braked.stages[index]
            roots = _all_roots(stage, index)
            restored = stage.output_angle
            nearest = min(
                roots,
                key=lambda root: abs(
                    root
                    - reference.stages[index].output_angle
                ),
            )
            assert nearest == pytest.approx(restored)


def test_restore_precision(mixer_definition):
    builder = MixedBrakeBuilder(mixer_definition)
    beta = math.radians(15)
    mechanism = builder.build(beta)
    for stage in mechanism.stages:
        error = abs(
            stage_error(
                stage,
                stage.input_angle,
                stage.output_angle,
            )
        )
        assert error < 1.0e-9


def test_unreachable_geometry_raises(mixer_definition):
    """For a large brake angle the moved mixer pivot is too
    far away for the frozen rod: no solution within the output
    bounds -> MechanismBuildError."""
    builder = MixedBrakeBuilder(mixer_definition)
    builder.build(0.0)
    builder.build(math.radians(10))
    with pytest.raises(MechanismBuildError):
        builder.build(math.radians(25))


def test_ambiguous_branch_restores_nearest():
    """Two-link planar chain where the frozen rod admits two
    solutions for the moved pivot; the builder must pick the
    one near the unbraked reference angle."""
    from model.lever_definition import LeverDefinition
    from model.mechanism_definition import (
        MechanismDefinition,
    )

    from core.point3d import Point3D
    from core.vector3d import Vector3D

    fixed = [
        LeverDefinition(
            id=1,
            pivot=Point3D(0.0, 0.0, 0.0),
            length_min=20,
            length_max=100,
            length_start=50,
            angle_min=math.radians(-60),
            angle_max=math.radians(60),
            angle_start=0.0,
            axis=Vector3D(0.0, 0.0, 1.0),
        ),
        LeverDefinition(
            id=2,
            pivot=Point3D(50.0, 0.0, 0.0),
            length_min=20,
            length_max=100,
            length_start=80,
            angle_min=math.radians(-90),
            angle_max=math.radians(90),
            angle_start=math.radians(30),
            axis=Vector3D(0.0, 0.0, 1.0),
            driver=1,
            pivot_on="lever1@0",
        ),
        LeverDefinition(
            id=3,
            pivot=Point3D(100.0, 40.0, 0.0),
            length_min=20,
            length_max=200,
            length_start=100,
            angle_min=math.radians(-180),
            angle_max=math.radians(180),
            angle_start=0.0,
            axis=Vector3D(0.0, 0.0, 1.0),
            driver=2,
        ),
    ]
    definition = MechanismDefinition(tuple(fixed))
    builder = MixedBrakeBuilder(definition)
    reference = builder.build(0.0)
    braked = builder.build(math.radians(20))
    assert braked.stages[1].rod_length == (
        pytest.approx(reference.stages[1].rod_length)
    )
    delta = abs(
        braked.stages[1].output_angle
        - reference.stages[1].output_angle
    )
    assert delta < math.radians(20)


def test_brake_angle_outside_bounds(mixer_definition):
    builder = MixedBrakeBuilder(mixer_definition)
    with pytest.raises(MechanismBuildError):
        builder.build(math.radians(90))


def test_reference_rod_lengths_not_cached_across_builders(
    mixer_definition,
):
    """Reference rod lengths are computed per builder
    instance, never globally cached across candidates."""
    builder_a = MixedBrakeBuilder(mixer_definition)
    reference_a = builder_a.build(0.0)

    from dataclasses import replace
    from model.mechanism_definition import (
        MechanismDefinition,
    )

    levers = [
        replace(
            lever,
            length_start=(
                lever.length_start + 10.0
            ),
        )
        for lever in mixer_definition.levers
    ]
    builder_b = MixedBrakeBuilder(
        MechanismDefinition(tuple(levers))
    )
    reference_b = builder_b.build(0.0)
    assert reference_b.stages[0].rod_length != (
        pytest.approx(reference_a.stages[0].rod_length)
    )


def test_mixer_requires_pivot_on(tmp_path):
    from model.lever_definition import LeverDefinition
    from model.mechanism_definition import (
        MechanismDefinition,
    )

    from core.point3d import Point3D
    from core.vector3d import Vector3D

    definition = MechanismDefinition(
        (
            LeverDefinition(
                id=1,
                pivot=Point3D(0.0, 0.0, 0.0),
                length_min=20,
                length_max=100,
                length_start=50,
                angle_min=-1.0,
                angle_max=1.0,
                angle_start=0.0,
                axis=Vector3D(0.0, 0.0, 1.0),
            ),
        )
    )
    with pytest.raises(MechanismBuildError):
        MixedBrakeBuilder(definition)


def test_pivot_on_requires_fixed_pivot_parent():
    """pivot_on may only point at a lever with a fixed
    pivot; pointing at another pivot_on lever is rejected
    by the topology check via the mixer-resolution rule."""
    from core.point3d import Point3D
    from core.vector3d import Vector3D
    from model.lever_definition import LeverDefinition
    from model.mechanism_definition import (
        MechanismDefinition,
    )

    definition = MechanismDefinition(
        (
            LeverDefinition(
                id=1,
                pivot=Point3D(0.0, 0.0, 0.0),
                length_min=20,
                length_max=100,
                length_start=50,
                angle_min=-1.0,
                angle_max=1.0,
                angle_start=0.0,
                axis=Vector3D(0.0, 0.0, 1.0),
                pivot_on="lever2@0",
            ),
            LeverDefinition(
                id=2,
                pivot=Point3D(50.0, 0.0, 0.0),
                length_min=20,
                length_max=100,
                length_start=50,
                angle_min=-1.0,
                angle_max=1.0,
                angle_start=0.0,
                axis=Vector3D(0.0, 0.0, 1.0),
                pivot_on="lever1@0",
            ),
        )
    )
    with pytest.raises(MechanismBuildError):
        MixedBrakeBuilder(definition)
