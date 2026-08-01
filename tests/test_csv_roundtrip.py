from __future__ import annotations

from mechanism_io import CsvReader, CsvWriter


def test_simulation_roundtrip(example_simulation_csv, tmp_path):
    original = CsvReader.read_simulation(example_simulation_csv)

    output = tmp_path / "simulation.csv"

    CsvWriter.write_simulation(original, output)

    restored = CsvReader.read_simulation(output)

    assert restored == original


def test_mechanism_roundtrip(example_mechanism_csv, tmp_path):
    original = CsvReader.read_mechanism(example_mechanism_csv)

    output = tmp_path / "mechanism.csv"

    CsvWriter.write_mechanism(original, output)

    restored = CsvReader.read_mechanism(output)

    assert restored == original