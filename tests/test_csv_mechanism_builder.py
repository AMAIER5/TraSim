from mechanics.csv_mechanism_builder import CsvMechanismBuilder
from mechanism_io.csv_reader import CsvReader


def test_build_creates_stages(example_mechanism_csv):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    mechanism = CsvMechanismBuilder().build(
        definition
    )

    assert len(mechanism.stages) == 3
    

def test_builder_preserves_lever_geometry(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    mechanism = CsvMechanismBuilder().build(
        definition
    )

    stage = mechanism.stages[1]

    assert stage.input_lever.pivot.x == 100
    assert stage.input_lever.pivot.y == 0

    assert stage.output_lever.pivot.x == 200
    assert stage.output_lever.pivot.y == 20
    
    
def test_builder_calculates_rod_length(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    mechanism = CsvMechanismBuilder().build(
        definition
    )

    stage = mechanism.stages[0]

    assert stage.rod_length == 85.0