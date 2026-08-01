"""
mechanism_io/csv_reader.py

Read simulation and mechanism definitions from CSV files.
"""

from __future__ import annotations

import csv
from pathlib import Path

from core.point3d import Point3D
from core.vector3d import Vector3D
from model.lever_definition import LeverDefinition
from model.mechanism_definition import MechanismDefinition
from model.simulation_config import SimulationConfig


class CsvReader:
    """
    CSV reader for simulation and mechanism definitions.
    """

    @staticmethod
    def read_simulation(
        path: str | Path,
    ) -> SimulationConfig:
        """
        Read simulation configuration from CSV.
        """

        values: dict[str, str] = {}

        with Path(path).open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                values[row["parameter"]] = row["value"]

        return SimulationConfig(
            population_size=int(
                values["population_size"]
            ),
            children_per_generation=int(
                values["children_per_generation"]
            ),
            generations=int(
                values["generations"]
            ),
            target_error=float(
                values["target_error"]
            ),
            mutation_rate=float(
                values["mutation_rate"]
            ),
            elite_size=int(
                values["elite_size"]
            ),
            motion_start=float(
                values["motion_start"]
            ),
            motion_end=float(
                values["motion_end"]
            ),
            motion_step=float(
                values["motion_step"]
            ),
        )

    @staticmethod
    def read_mechanism(
        path: str | Path,
    ) -> MechanismDefinition:
        """
        Read mechanism definition from CSV.
        """

        levers: list[LeverDefinition] = []

        with Path(path).open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                levers.append(
                    CsvReader._parse_lever(row)
                )

        return MechanismDefinition(
            tuple(levers)
        )

    @staticmethod
    def _parse_lever(
        row: dict[str, str],
    ) -> LeverDefinition:
        """
        Parse one lever definition.
        """

        coupled = CsvReader._parse_optional_int(
            row["coupled"]
        )

        driver = CsvReader._parse_optional_int(
            row["driver"]
        )

        # coupled has priority over driver
        if coupled is not None:
            driver = None

        return LeverDefinition(
            id=int(row["id"]),

            length_min=float(
                row["length_min"]
            ),
            length_max=float(
                row["length_max"]
            ),
            length_start=float(
                row["length_start"]
            ),

            angle_min=float(
                row["angle_min"]
            ),
            angle_max=float(
                row["angle_max"]
            ),
            angle_start=float(
                row["angle_start"]
            ),

            pivot=Point3D(
                x=float(row["pivot_x"]),
                y=float(row["pivot_y"]),
                z=float(row["pivot_z"]),
            ),

            axis=Vector3D(
                x=float(row["axis_x"]),
                y=float(row["axis_y"]),
                z=float(row["axis_z"]),
            ),

            driver=driver,
            coupled=coupled,
        )

    @staticmethod
    def _parse_optional_int(
        value: str | None,
    ) -> int | None:
        """
        Parse optional integer values.

        Empty CSV cells become None.
        """

        if value is None or value.strip() == "":
            return None

        return int(value)