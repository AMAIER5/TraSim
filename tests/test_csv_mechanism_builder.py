from mechanics.csv_mechanism_builder import CsvMechanismBuilder
from mechanism_io.csv_reader import CsvReader
from optimization.parameter_set import ParameterSet


def test_build_creates_stages(example_mechanism_csv):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    mechanism = CsvMechanismBuilder(
        definition
    ).build(
        ParameterSet(())
    )

    assert len(mechanism.stages) == 3


def test_builder_preserves_lever_geometry(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    mechanism = CsvMechanismBuilder(
        definition
    ).build(
        ParameterSet(())
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

    mechanism = CsvMechanismBuilder(
        definition
    ).build(
        ParameterSet(())
    )

    stage = mechanism.stages[0]

    assert stage.rod_length == 85.0
    
def test_builder_preserves_stage_order(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    mechanism = CsvMechanismBuilder(
        definition
    ).build(
        ParameterSet(())
    )

    assert len(mechanism.stages) == 3

    stage1 = mechanism.stages[0]
    stage2 = mechanism.stages[1]
    stage3 = mechanism.stages[2]

    # Stage 1: Lever 1 -> Lever 2
    assert stage1.input_lever.pivot.x == 0
    assert stage1.output_lever.pivot.x == 100

    # Stage 2: Lever 2 -> Lever 3
    assert stage2.input_lever.pivot.x == 100
    assert stage2.output_lever.pivot.x == 200

    # Stage 3: Lever 3 -> Lever 4
    assert stage3.input_lever.pivot.x == 200
    assert stage3.output_lever.pivot.x == 300
    
def test_builder_validates_created_stages(
    example_mechanism_csv,
):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv,
    )

    builder = CsvMechanismBuilder(
        definition
    )

    mechanism = builder.build(
        ParameterSet(())
    )

    results = builder.get_validation_results()

    assert len(results) == len(
        mechanism.stages
    )

    assert all(
        result.stage_id is not None
        for result in results
    )