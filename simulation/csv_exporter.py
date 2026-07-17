"""
simulation/csv_exporter.py

CSV export for simulation results.
"""

from __future__ import annotations

import csv
from pathlib import Path

from simulation.mechanism_motion_simulator import (
    MechanismMotionResult,
)


class CSVExporter:
    """
    Writes mechanism simulation results
    into CSV format.

    Angles are exported in radians.
    """

    def write(
        self,
        filename: str | Path,
        result: MechanismMotionResult,
    ) -> None:
        """
        Write simulation result.
        """

        filename = Path(filename)

        with filename.open(
            "w",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                self._create_header(
                    result
                )
            )

            for index, angle in enumerate(
                result.input_angles
            ):

                row = [
                    angle
                ]

                row.extend(
                    result.stage_outputs[index]
                )

                writer.writerow(
                    row
                )

    @staticmethod
    def _create_header(
        result: MechanismMotionResult,
    ) -> list[str]:
        """
        Create CSV header.
        """

        if not result.stage_outputs:

            return [
                "input_angle"
            ]

        stage_count = len(
            result.stage_outputs[0]
        )

        return (
            [
                "input_angle"
            ]
            +
            [
                f"stage_{index}_output"
                for index in range(stage_count)
            ]
        )