from __future__ import annotations

import math

import pytest

from mechanism_io import CsvReader

from core.vector3d import Vector3D

def test_read_simulation(example_simulation_csv):
    config = CsvReader.read_simulation(example_simulation_csv)

    assert config.population_size == 200
    assert config.children_per_generation == 50
    assert config.generations == 500
    assert config.target_error == 0.05
    assert config.mutation_rate == 0.15
    assert config.elite_size == 5
    # Angles are now in radians (converted from degrees in CSV)
    assert math.isclose(config.motion_start, math.radians(-50))
    assert math.isclose(config.motion_end, math.radians(50))
    assert math.isclose(config.motion_step, math.radians(1))

def test_read_mechanism(example_mechanism_csv):
    mechanism = CsvReader.read_mechanism(example_mechanism_csv)

    assert mechanism.lever_count == 4

    lever = mechanism.get_lever(2)

    assert lever.length_start == 45

    assert lever.pivot.x == 100
    assert lever.pivot.y == 0
    assert lever.pivot.z == 0

    assert lever.axis.x == 0
    assert lever.axis.y == 0
    assert lever.axis.z == 1

    assert lever.driver == 1
    assert lever.coupled is None

def test_driver_ignored_when_coupled(example_coupled_csv):
    mechanism = CsvReader.read_mechanism(example_coupled_csv)

    lever = mechanism.get_lever(3)

    assert lever.coupled == 2
    assert lever.driver is None

def test_read_lever_geometry(example_mechanism_csv):
    mechanism = CsvReader.read_mechanism(example_mechanism_csv)

    lever = mechanism.get_lever(3)

    assert lever.pivot.x == 200
    assert lever.pivot.y == 20
    assert lever.pivot.z == 0

    assert lever.axis == Vector3D(
        0,
        0,
        1,
    )


def test_read_lever_without_reference_direction_is_none(
    example_mechanism_csv,
):
    mechanism = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    for lever in mechanism.levers:
        assert lever.reference_direction is None


def test_read_lever_with_explicit_reference_direction(
    tmp_path,
):
    path = tmp_path / "mechanism.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,ref_x,ref_y,ref_z,driver,coupled
1,40,100,60,-40,40,0,0,0,0,0,0,1,1,0,0,,
2,30,90,45,-60,60,0,100,0,0,0,0,1,0,1,0,1,
""",
        encoding="utf-8",
    )

    mechanism = CsvReader.read_mechanism(path)

    assert mechanism.get_lever(1).reference_direction == Vector3D(1, 0, 0)
    assert mechanism.get_lever(2).reference_direction == Vector3D(0, 1, 0)


def test_partial_reference_direction_raises(
    tmp_path,
):
    path = tmp_path / "mechanism.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,ref_x,ref_y,ref_z,driver,coupled
1,40,100,60,-40,40,0,0,0,0,0,0,1,1,0,,,
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ref_x"):
        CsvReader.read_mechanism(path)