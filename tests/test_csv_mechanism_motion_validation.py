from mechanics.csv_mechanism_builder import CsvMechanismBuilder
from mechanism_io.csv_reader import CsvReader
from optimization.parameter_set import ParameterSet
from tests.conftest import example_mechanism_csv
from validation.stage_motion_validator import StageMotionValidator


def test_build_diagnosis(example_mechanism_csv):
    definition = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    mechanism = CsvMechanismBuilder(
        definition
    ).build(
        ParameterSet(())
    )

    validator = StageMotionValidator(
        steps=100,
    )

    diagnostics = []

    for index, stage in enumerate(
        mechanism.stages,
        start=1,
    ):
        result = validator.validate(
            stage,
            stage_id=index,
        )

        diagnostics.append(
            f"""
Stage {index}
-----------
valid={result.valid}
checked_steps={result.checked_steps}
failed_at={result.failed_at_input_angle}
reason={result.reason}
"""
        )

    assert False, "\n".join(diagnostics)
    

def test_stage1_residual_scan(example_mechanism_csv):

    import math

    from solver.objective import create_stage_objective

    definition = CsvReader.read_mechanism(
        example_mechanism_csv
    )

    mechanism = CsvMechanismBuilder(
        definition
    ).build(
        ParameterSet(())
    )

    stage = mechanism.stages[0]

    residual = create_stage_objective(
        stage,
        math.radians(-40),
    )

    print("\nResidual scan Stage 1 @ input -40°")

    for deg in range(-180, 181, 10):

        value = residual(
            math.radians(deg)
        )

        print(
            deg,
            value,
        )