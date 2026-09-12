"""
mechanism_io/csv_reader.py

Read simulation and mechanism definitions from CSV files.

Angles in CSV are in DEGREES and are automatically converted to radians.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from core.point3d import Point3D
from core.vector3d import Vector3D
from model.lever_definition import LeverDefinition
from model.mechanism_definition import MechanismDefinition
from model.simulation_config import SimulationConfig

class CsvReader:
    """
    CSV reader for simulation and mechanism definitions.

    Note: All angle values in CSV files are interpreted as DEGREES
    and automatically converted to radians internally.
    """

    @staticmethod
    def read_simulation(
        path: str | Path,
    ) -> SimulationConfig:
        """
        Read simulation configuration from CSV.

        All angle values (motion_start, motion_end, motion_step) are in DEGREES.
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

        # Convert angle values from degrees to radians
        motion_start = math.radians(float(values["motion_start"]))
        motion_end = math.radians(float(values["motion_end"]))
        motion_step = math.radians(float(values["motion_step"]))

        return SimulationConfig(
            population_size=int(values["population_size"]),
            children_per_generation=int(values["children_per_generation"]),
            generations=int(values["generations"]),
            target_error=float(values["target_error"]),
            mutation_rate=float(values["mutation_rate"]),
            elite_size=int(values["elite_size"]),
            motion_start=motion_start,
            motion_end=motion_end,
            motion_step=motion_step,
        )

    @staticmethod
    def read_mechanism(
        path: str | Path,
    ) -> MechanismDefinition:
        """
        Read mechanism definition from CSV.

        All angle values (angle_min, angle_max, angle_start) are in DEGREES.
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

        All angle values are converted from DEGREES to RADIANS.
        """
        coupled = CsvReader._parse_optional_int(row.get("coupled", ""))
        driver = CsvReader._parse_optional_int(row.get("driver", ""))

        # Coupled has priority over driver
        if coupled is not None:
            driver = None

        reference_direction = (
            CsvReader._parse_optional_vector(row)
        )

        return LeverDefinition(
            id=int(row["id"]),

            length_min=float(row["length_min"]),
            length_max=float(row["length_max"]),
            length_start=float(row["length_start"]),

            # Convert angles from degrees to radians.
            # Angles are lever angles measured relative to the
            # lever's reference_direction about its axis.
            angle_min=math.radians(float(row["angle_min"])),
            angle_max=math.radians(float(row["angle_max"])),
            angle_start=math.radians(float(row["angle_start"])),

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

            reference_direction=reference_direction,

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

    @staticmethod
    def _parse_optional_vector(
        row: dict[str, str],
    ) -> Vector3D | None:
        """
        Parse an optional reference direction from the
        ``ref_x``/``ref_y``/``ref_z`` columns.

        When all three columns are absent or empty the lever
        selects its reference direction automatically.
        """
        keys = ("ref_x", "ref_y", "ref_z")

        present = {
            key: row.get(key)
            for key in keys
            if row.get(key) is not None
            and row.get(key, "").strip() != ""
        }

        if not present:
            return None

        if len(present) != len(keys):
            raise ValueError(
                "reference direction requires all "
                "of ref_x, ref_y, ref_z."
            )

        return Vector3D(
            x=float(present["ref_x"]),
            y=float(present["ref_y"]),
            z=float(present["ref_z"]),
        )