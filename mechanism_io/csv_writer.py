"""
mechanism_io/csv_writer.py

Write simulation and mechanism definitions to CSV files.
"""

from __future__ import annotations

import csv
from pathlib import Path

from model.mechanism_definition import MechanismDefinition
from model.simulation_config import SimulationConfig


class CsvWriter:
    """
    CSV writer for simulation and mechanism definitions.
    """

    @staticmethod
    def write_simulation(
        config: SimulationConfig,
        path: str | Path,
    ) -> None:
        """
        Write simulation configuration to CSV.
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
            writer.writerow(
                (
                    "motion_start",
                    config.motion_start,
                )
            )
            writer.writerow(
                (
                    "motion_end",
                    config.motion_end,
                )
            )
            writer.writerow(
                (
                    "motion_step",
                    config.motion_step,
                )
            )

    @staticmethod
    def write_mechanism(
        mechanism: MechanismDefinition,
        path: str | Path,
    ) -> None:
        """
        Write mechanism definition to CSV.
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
                writer.writerow(
                    (
                        lever.id,

                        lever.length_min,
                        lever.length_max,
                        lever.length_start,

                        lever.angle_min,
                        lever.angle_max,
                        lever.angle_start,

                        lever.pivot.x,
                        lever.pivot.y,
                        lever.pivot.z,

                        lever.axis.x,
                        lever.axis.y,
                        lever.axis.z,

                        "" if lever.driver is None
                        else lever.driver,

                        "" if lever.coupled is None
                        else lever.coupled,
                    )
                )