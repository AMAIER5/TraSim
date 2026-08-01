from __future__ import annotations

from mechanism_io import CsvReader, CsvWriter


def test_write_simulation(example_simulation_csv, tmp_path):
    config = CsvReader.read_simulation(example_simulation_csv)

    output = tmp_path / "simulation.csv"

    CsvWriter.write_simulation(config, output)

    loaded = CsvReader.read_simulation(output)

    assert loaded == config


def test_write_mechanism(example_mechanism_csv, tmp_path):
    mechanism = CsvReader.read_mechanism(example_mechanism_csv)

    output = tmp_path / "mechanism.csv"

    CsvWriter.write_mechanism(mechanism, output)

    loaded = CsvReader.read_mechanism(output)

    assert loaded == mechanism


def test_write_mechanism_contains_geometry(
    example_mechanism_csv,
    tmp_path,
):
    mechanism = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    output = tmp_path / "mechanism.csv"

    CsvWriter.write_mechanism(
        mechanism,
        output,
    )

    content = output.read_text(
        encoding="utf-8"
    )

    assert "pivot_x" in content
    assert "axis_z" in content
    assert "coupled" in content