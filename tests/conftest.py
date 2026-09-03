"""
tests/conftest.py
"""

from __future__ import annotations

import pytest

@pytest.fixture
def example_simulation_csv(tmp_path):
    """
    Create an example simulation CSV file.
    All angle values are in DEGREES.
    """
    path = tmp_path / "simulation.csv"
    path.write_text(
        """parameter,value
population_size,200
children_per_generation,50
generations,500
target_error,0.05
mutation_rate,0.15
elite_size,5
motion_start,-50
motion_end,50
motion_step,1
""",
        encoding="utf-8",
    )
    return path

@pytest.fixture
def example_mechanism_csv(tmp_path):
    """
    Create an example mechanism CSV file.
    All angle values are in DEGREES.
    Original geometry: lever 3 and 4 at y=20
    """
    path = tmp_path / "mechanism.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,40,100,60,-40,40,0,0,0,0,0,0,1,,
2,30,90,45,-60,60,0,100,0,0,0,0,1,1,
3,20,70,35,-30,30,0,200,20,0,0,0,1,2,
4,20,80,40,-45,45,10,300,20,0,0,0,1,3,
""",
        encoding="utf-8",
    )
    return path

@pytest.fixture
def example_coupled_csv(tmp_path):
    """
    Create a mechanism containing a coupled lever.
    All angle values are in DEGREES.
    """
    path = tmp_path / "coupled.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,40,100,60,-40,40,0,0,0,0,0,0,1,,
2,30,90,45,-60,60,0,100,0,0,0,0,1,1,
3,20,70,35,-30,30,0,200,20,0,0,0,1,,2
""",
        encoding="utf-8",
    )
    return path

@pytest.fixture
def simple_stage_csv(tmp_path):
    """
    Create a simple two lever mechanism.
    All angle values are in DEGREES.
    """
    path = tmp_path / "simple_stage.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,30,50,40,-60,60,0,0,0,0,0,0,1,,
2,90,110,100,-60,60,0,50,-10,0,0,0,1,1,
""",
        encoding="utf-8",
    )
    return path

@pytest.fixture
def simple_target_csv(tmp_path):
    """
    Create a simple target transfer curve.
    All angle values are in DEGREES.
    """
    path = tmp_path / "target.csv"
    path.write_text(
        """input_angle,output_angle
-40,-30
-20,-15
0,0
20,15
40,30
""",
        encoding="utf-8",
    )
    return path

@pytest.fixture
def simple_multistage_csv(tmp_path):
    """
    All angle values are in DEGREES.
    """
    path = tmp_path / "multistage.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,50,50,50,-90,90,0,0,0,0,0,0,1,,
2,50,50,50,-90,90,0,100,0,0,0,0,1,1,
3,50,50,50,-90,90,0,200,0,0,0,0,1,2,
""",
        encoding="utf-8",
    )
    return path

@pytest.fixture
def example_mechanism_colinear_csv(tmp_path):
    """
    Create a colinear mechanism CSV for tests requiring robust kinematics.
    All pivots on X-axis (y=0, z=0), all axes Z-axis.
    All angle values are in DEGREES.
    """
    path = tmp_path / "mechanism.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,40,100,60,-40,40,0,0,0,0,0,0,1,,
2,30,90,45,-60,60,0,100,0,0,0,0,1,1,
3,20,70,35,-30,30,0,200,0,0,0,0,1,2,
4,20,80,40,-45,45,0,300,0,0,0,0,1,3,
""",
        encoding="utf-8",
    )
    return path

@pytest.fixture
def example_mechanism_2stage_csv(tmp_path):
    """
    Create a simple 2-stage mechanism for parameter influence tests.
    All pivots on X-axis, all axes Z-axis.
    All angle values are in DEGREES.
    """
    path = tmp_path / "mechanism.csv"
    path.write_text(
        """id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,40,100,60,-90,90,0,0,0,0,0,0,1,,
2,30,90,45,-90,90,0,100,0,0,0,0,1,1,
3,20,70,35,-90,90,0,200,0,0,0,0,1,2,
""",
        encoding="utf-8",
    )
    return path