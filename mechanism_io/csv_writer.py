"""
mechanism_io/csv_writer.py

Write simulation and mechanism definitions to CSV files.

Angles are written in DEGREES (converted from internal radians).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from mechanism_io.csv_convention import (
    CsvConvention,
    INTERNATIONAL,
)
from model.mechanism_definition import MechanismDefinition
from model.simulation_config import SimulationConfig

class CsvWriter:
    """
    CSV writer for simulation and mechanism definitions.

    Note: All angle values are written as DEGREES.

    By default files are written in the international
    convention (comma as column separator, dot as
    decimal separator).  Pass ``convention=GERMAN``
    to write files in the German convention
    (semicolon as column separator, comma as decimal
    separator).
    """

    @staticmethod
    def write_simulation(
        config: SimulationConfig,
        path: str | Path,
        *,
        convention: CsvConvention = INTERNATIONAL,
    ) -> None:
        """
        Write simulation configuration to CSV.

        All angle values (motion_start, motion_end,
        motion_step) are written in DEGREES.
        """
        with Path(path).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(
                file,
                delimiter=convention.delimiter,
            )

            writer.writerow(
                (
                    "parameter",
                    "value",
                )
            )

            # Convert radians back to degrees for CSV
            writer.writerow(
                (
                    "population_size",
                    convention.format_float(
                        config.population_size
                    ),
                )
            )
            writer.writerow(
                (
                    "children_per_generation",
                    convention.format_float(
                        config.children_per_generation,
                    ),
                )
            )
            writer.writerow(
                (
                    "generations",
                    convention.format_float(
                        config.generations
                    ),
                )
            )
            writer.writerow(
                (
                    "target_error",
                    convention.format_float(
                        config.target_error
                    ),
                )
            )
            writer.writerow(
                (
                    "mutation_rate",
                    convention.format_float(
                        config.mutation_rate
                    ),
                )
            )
            writer.writerow(
                (
                    "elite_size",
                    convention.format_float(
                        config.elite_size
                    ),
                )
            )
            # Convert motion angles from radians to degrees
            writer.writerow(
                (
                    "motion_start",
                    convention.format_float(
                        math.degrees(
                            config.motion_start
                        )
                    ),
                )
            )
            writer.writerow(
                (
                    "motion_end",
                    convention.format_float(
                        math.degrees(
                            config.motion_end
                        )
                    ),
                )
            )
            writer.writerow(
                (
                    "motion_step",
                    convention.format_float(
                        math.degrees(
                            config.motion_step
                        )
                    ),
                )
            )

    @staticmethod
    def write_mechanism(
        mechanism: MechanismDefinition,
        path: str | Path,
        *,
        convention: CsvConvention = INTERNATIONAL,
    ) -> None:
        """
        Write mechanism definition to CSV.

        All angle values (angle_min, angle_max,
        angle_start) are written in DEGREES.
        """
        with Path(path).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(
                file,
                delimiter=convention.delimiter,
            )

            writer.writerow(
                (
                    "id",
                    "length_min",
                    "length_max",
                    "length_start",
                    "angle_min",
                    "angle_max",
                    "angle_start",
                    "pivot_x",
                    "pivot_y",
                    "pivot_z",
                    "axis_x",
                    "axis_y",
                    "axis_z",
                    "ref_x",
                    "ref_y",
                    "ref_z",
                    "driver",
                    "coupled",
                )
            )

            for lever in mechanism.levers:
                # Convert angle values from radians to degrees.
                # Angles are lever angles measured relative to the
                # lever's reference_direction about its axis.
                ref = lever.reference_direction
                writer.writerow(
                    (
                        lever.id,
                        convention.format_float(
                            lever.length_min
                        ),
                        convention.format_float(
                            lever.length_max
                        ),
                        convention.format_float(
                            lever.length_start
                        ),
                        convention.format_float(
                            math.degrees(
                                lever.angle_min
                            )
                        ),
                        convention.format_float(
                            math.degrees(
                                lever.angle_max
                            )
                        ),
                        convention.format_float(
                            math.degrees(
                                lever.angle_start
                            )
                        ),
                        convention.format_float(
                            lever.pivot.x
                        ),
                        convention.format_float(
                            lever.pivot.y
                        ),
                        convention.format_float(
                            lever.pivot.z
                        ),
                        convention.format_float(
                            lever.axis.x
                        ),
                        convention.format_float(
                            lever.axis.y
                        ),
                        convention.format_float(
                            lever.axis.z
                        ),
                        "" if ref is None
                        else convention.format_float(
                            ref.x
                        ),
                        "" if ref is None
                        else convention.format_float(
                            ref.y
                        ),
                        "" if ref is None
                        else convention.format_float(
                            ref.z
                        ),
                        "" if lever.driver is None
                        else lever.driver,
                        "" if lever.coupled is None
                        else lever.coupled,
                    )
                )