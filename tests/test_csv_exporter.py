"""
tests/test_csv_exporter.py

Tests for CSV export.
"""

from __future__ import annotations

import csv

from simulation.csv_exporter import CSVExporter
from simulation.mechanism_motion_simulator import (
    MechanismMotionResult,
)


def create_result():

    return MechanismMotionResult(
        input_angles=(
            0.0,
            1.0,
        ),
        stage_outputs=(
            (
                0.1,
                0.2,
            ),
            (
                1.1,
                1.2,
            ),
        ),
        success=True,
    )


def test_csv_export_creates_file(
    tmp_path,
):

    filename = (
        tmp_path
        /
        "simulation.csv"
    )

    exporter = CSVExporter()

    exporter.write(
        filename,
        create_result(),
    )

    assert filename.exists()


def test_csv_contains_header(
    tmp_path,
):

    filename = (
        tmp_path
        /
        "simulation.csv"
    )

    exporter = CSVExporter()

    exporter.write(
        filename,
        create_result(),
    )

    with open(
        filename,
        newline="",
    ) as file:

        rows = list(
            csv.reader(file)
        )

    assert rows[0] == [
        "input_angle",
        "stage_0_output",
        "stage_1_output",
    ]


def test_csv_contains_values(
    tmp_path,
):

    filename = (
        tmp_path
        /
        "simulation.csv"
    )

    exporter = CSVExporter()

    exporter.write(
        filename,
        create_result(),
    )

    with open(
        filename,
        newline="",
    ) as file:

        rows = list(
            csv.reader(file)
        )

    assert rows[1][0] == "0.0"

    assert rows[1][1] == "0.1"

    assert rows[1][2] == "0.2"