"""
tests/test_fast_residual.py

Equivalence tests for the allocation-free solver hot path.

The scalar fast path (Lever.end_position_components,
solver._rod_residual) must be bit-identical to the object-based
path (Lever.end_position + solver.constraints.rod_length_error),
whether the pure-Python module or the Cython-transpiled native
extension is active.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from solver._rod_residual import (
    make_stage_residual,
    rod_length_residual,
    stage_residual,
)
from solver.constraints import rod_length_error
from solver.objective import (
    create_stage_objective,
    stage_error,
)


def make_planar_lever() -> Lever:

    return Lever(
        pivot=Point3D(120.0, 5.0, -3.0),
        axis=Vector3D(0.0, 0.0, 1.0),
        length=60.0,
        reference_direction=Vector3D(1.0, 0.0, 0.0),
    )


def make_spatial_lever() -> Lever:

    axis = Vector3D(1.0, 2.0, 3.0)

    return Lever(
        pivot=Point3D(-40.0, 25.0, 11.0),
        axis=axis,
        length=55.0,
        reference_direction=Vector3D(2.0, -1.0, 0.0),
    )


def make_stage(
    output_lever: Lever,
) -> Stage:

    input_lever = Lever(
        pivot=Point3D(0.0, 0.0, 0.0),
        axis=Vector3D(0.0, 0.0, 1.0),
        length=70.0,
        reference_direction=Vector3D(1.0, 0.0, 0.0),
    )

    return Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=math.radians(-20.0),
        output_angle=math.radians(35.0),
    )


def angle_sweep() -> list[float]:

    return [
        math.radians(degree)
        for degree in range(-360, 361, 7)
    ]


# ---------------------------------------------------------------------------
# rod_length_residual vs rod_length_error
# ---------------------------------------------------------------------------

def test_rod_length_residual_bit_identical_to_object_path():

    for degree in range(-360, 361, 13):

        angle = math.radians(degree)

        point_a = Point3D(
            10.0 + degree,
            -20.0 + 0.5 * degree,
            5.0,
        )

        point_b = Point3D(
            100.0 * math.cos(angle),
            100.0 * math.sin(angle),
            7.5,
        )

        expected = rod_length_error(
            point_a,
            point_b,
            123.456,
        )

        actual = rod_length_residual(
            point_b.x,
            point_b.y,
            point_b.z,
            point_a.x,
            point_a.y,
            point_a.z,
            123.456,
        )

        assert actual == expected


# ---------------------------------------------------------------------------
# Lever.end_position_components vs Lever.end_position
# ---------------------------------------------------------------------------

def test_end_position_components_bit_identical_to_end_position():

    for lever in (
        make_planar_lever(),
        make_spatial_lever(),
    ):

        for angle in angle_sweep():

            point = lever.end_position(angle)

            components = lever.end_position_components(
                angle
            )

            assert components == (
                point.x,
                point.y,
                point.z,
            )


# ---------------------------------------------------------------------------
# stage_residual vs object path
# ---------------------------------------------------------------------------

def test_stage_residual_bit_identical_to_object_path():

    for lever in (
        make_planar_lever(),
        make_spatial_lever(),
    ):

        stage = make_stage(lever)

        constants = lever.endpoint_constants()

        input_point = stage.input_position(
            stage.input_angle
        )

        for angle in angle_sweep():

            output_point = stage.output_position(angle)

            expected = rod_length_error(
                input_point,
                output_point,
                stage.rod_length,
            )

            actual = stage_residual(
                angle,
                *constants,
                input_point.x,
                input_point.y,
                input_point.z,
                stage.rod_length,
            )

            assert actual == expected


# ---------------------------------------------------------------------------
# make_stage_residual closure vs stage_error
# ---------------------------------------------------------------------------

def test_make_stage_residual_bit_identical_to_stage_error():

    for lever in (
        make_planar_lever(),
        make_spatial_lever(),
    ):

        stage = make_stage(lever)

        residual = make_stage_residual(
            lever.endpoint_constants(),
            stage.input_endpoint.x,
            stage.input_endpoint.y,
            stage.input_endpoint.z,
            stage.rod_length,
        )

        for angle in angle_sweep():

            assert residual(angle) == stage_error(
                stage,
                stage.input_angle,
                angle,
            )


# ---------------------------------------------------------------------------
# create_stage_objective (with and without cached constants)
# ---------------------------------------------------------------------------

def test_create_stage_objective_bit_identical_to_stage_error():

    stage = make_stage(
        make_spatial_lever()
    )

    residual = create_stage_objective(
        stage,
        stage.input_angle,
    )

    for angle in angle_sweep():

        assert residual(angle) == stage_error(
            stage,
            stage.input_angle,
            angle,
        )


def test_create_stage_objective_with_cached_constants_matches_default():

    stage = make_stage(
        make_planar_lever()
    )

    constants = stage.output_lever.endpoint_constants()

    default_residual = create_stage_objective(
        stage,
        stage.input_angle,
    )

    cached_residual = create_stage_objective(
        stage,
        stage.input_angle,
        output_constants=constants,
    )

    for angle in angle_sweep():

        assert (
            cached_residual(angle)
            == default_residual(angle)
        )
