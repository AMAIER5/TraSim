"""
mechanism_io/csv_writer.py

Write simulation and mechanism definitions to CSV files.

Angles are written in DEGREES (converted from internal radians).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from model.mechanism_definition import MechanismDefinition
from model.simulation_config import SimulationConfig

class CsvWriter:
    """
    CSV writer for simulation and mechanism definitions.

    Note: All angle values are written as DEGREES.
    """

    @staticmethod
    def write_simulation(
        config: SimulationConfig,
        path: str | Path,
    ) -> None:
        """
        Write simulation configuration to CSV.

        All angle values (motion_start, motion_end, motion_step) are written in DEGREES.
        """
        with Path(path).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

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
                    config.population_size,
                )
            )
            writer.writerow(
                (
                    "children_per_generation",
                    config.children_per_generation,
                )
            )
            writer.writerow(
                (
                    "generations",
                    config.generations,
                )
            )
            writer.writerow(
                (
                    "target_error",
                    config.target_error,
                )
            )
            writer.writerow(
                (
                    "mutation_rate",
                    config.mutation_rate,
                )
            )
            writer.writerow(
                (
                    "elite_size",
                    config.elite_size,
                )
            )
            # Convert motion angles from radians to degrees
            writer.writerow(
                (
                    "motion_start",
                    math.degrees(config.motion_start),
                )
            )
            writer.writerow(
                (
                    "motion_end",
                    math.degrees(config.motion_end),
                )
            )
            writer.writerow(
                (
                    "motion_step",
                    math.degrees(config.motion_step),
                )
            )

    @staticmethod
    def write_mechanism(
        mechanism: MechanismDefinition,
        path: str | Path,
    ) -> None:
        """
        Write mechanism definition to CSV.

        All angle values (angle_min, angle_max, angle_start) are written in DEGREES.
        """
        with Path(path).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

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
                    "driver",
                    "coupled",
                )
            )

            for lever in mechanism.levers:
                # Convert angle values from radians to degrees
                writer.writerow(
                    (
                        lever.id,
                        lever.length_min,
                        lever.length_max,
                        lever.length_start,
                        math.degrees(lever.angle_min),
                        math.degrees(lever.angle_max),
                        math.degrees(lever.angle_start),
                        lever.pivot.x,
                        lever.pivot.y,
                        lever.pivot.z,
                        lever.axis.x,
                        lever.axis.y,
                        lever.axis.z,
                        "" if lever.driver is None else lever.driver,
                        "" if lever.coupled is None else lever.coupled,
                    )
                )