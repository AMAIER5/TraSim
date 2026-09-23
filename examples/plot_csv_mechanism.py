"""examples/plot_csv_mechanism.py

Visualization of a simulation result as an HTML document.

The script loads a mechanism from mechanism.csv,
simulates it against target_curve.csv and writes an
HTML document (simulation_result.html) with two
diagrams:

1. A three-dimensional diagram showing every lever as
   a line in three positions: at the minimum, the
   middle and the maximum drive angle of the first
   (driving) lever.
2. A two-dimensional diagram showing the input curve,
   the desired output curve (Soll) and the achieved
   output curve (Ist).

The mechanism is built from the CSV reference values
(length_start / angle_start); no optimization is run.
Set OPTIMIZE = True to optimize the mechanism first
(see optimize_csv_mechanism.py for the optimization
setup).

User I/O angles are in DEGREES.
Internal calculations use RADIANS.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from analysis.curve_plotter import (
    CurvePlotter,
    mechanism_states_from_results,
)
from analysis.target_curve import TargetCurve
from mechanism_io.csv_reader import CsvReader
from mechanics.csv_mechanism_builder import (
    CsvMechanismBuilder,
)
from optimization.parameter import Parameter
from optimization.parameter_set import (
    ParameterSet,
)
from simulation.mechanism_simulator import (
    MechanismSimulator,
)
from simulation.point_motion import PointMotion
from simulation.stage_simulator import (
    StageSimulator,
)

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

MECHANISM_FILE = BASE_DIR / "mechanism.csv"
TARGET_FILE = BASE_DIR / "target_curve.csv"
OUTPUT_FILE = (
    BASE_DIR / "simulation_result.html"
)

# -------------------------------------------------
# Mechanism definition
# -------------------------------------------------

definition = CsvReader.read_mechanism(
    MECHANISM_FILE,
)
builder = CsvMechanismBuilder(definition)

parameters = []
for lever in definition.levers:
    parameters.append(
        Parameter(
            name=f"lever.{lever.id}.length",
            value=lever.length_start,
            minimum=lever.length_min,
            maximum=lever.length_max,
        ),
    )
    parameters.append(
        Parameter(
            name=f"lever.{lever.id}.angle",
            value=lever.angle_start,
            minimum=lever.angle_min,
            maximum=lever.angle_max,
        ),
    )
parameter_set = ParameterSet(
    parameters=tuple(parameters),
)
mechanism = builder.build(parameter_set)

# -------------------------------------------------
# Target curve
# -------------------------------------------------

target_input_angles_deg = []
target_output_angles_deg = []
with open(
    TARGET_FILE,
    newline="",
    encoding="utf-8",
) as file:
    reader = csv.DictReader(file)
    for row in reader:
        target_input_angles_deg.append(
            float(row["input_angle"]),
        )
        target_output_angles_deg.append(
            float(row["output_angle"]),
        )
target_input_angles_rad = tuple(
    math.radians(angle)
    for angle in target_input_angles_deg
)
target_output_angles_rad = tuple(
    math.radians(angle)
    for angle in target_output_angles_deg
)

# -------------------------------------------------
# Simulation
# -------------------------------------------------

motion = PointMotion(
    angles=target_input_angles_rad,
)
simulator = MechanismSimulator(
    motion=motion,
    stage_simulator=StageSimulator(),
    stage_limit=None,
)
simulation_results = simulator.simulate(
    mechanism,
)

final_result = simulation_results[-1]
drive_result = simulation_results[0]
all_succeeded = all(
    result.success
    for result in simulation_results
)
if not all_succeeded:
    blocked = next(
        (
            result.blocked_at
            for result in simulation_results
            if not result.success
        ),
        None,
    )
    raise RuntimeError(
        "Simulation blocked at "
        f"{math.degrees(blocked):.1f} deg; "
        "no complete result to plot."
    )

# -------------------------------------------------
# Plot data
# -------------------------------------------------

plotter = CurvePlotter(
    title=(
        "TraSim \u2014 Simulationsergebnis "
        f"({MECHANISM_FILE.name})"
    ),
)

plotter.set_input_curve(
    drive_result.input_angles,
)

plotter.set_target_curve(
    TargetCurve.from_points(
        target_input_angles_rad,
        target_output_angles_rad,
    ),
    input_angles=target_input_angles_rad,
    output_angles=target_output_angles_rad,
)

plotter.add_actual_curve(
    drive_result.input_angles,
    final_result.output_angles,
    label="Ist-Ausgangskurve",
)

for state in mechanism_states_from_results(
    mechanism,
    simulation_results,
):
    plotter.add_mechanism_state(state)

# -------------------------------------------------
# Write HTML document
# -------------------------------------------------

output_path = plotter.write(OUTPUT_FILE)

print("=" * 80)
print("VISUALIZATION")
print("=" * 80)
print(f"\nInput curve : {len(drive_result.input_angles)} points")
print(f"Output curve: {len(final_result.output_angles)} points")
print(f"\nHTML document written to: {output_path}")
