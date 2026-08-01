"""
model/mechanism_definition.py

Definition of a complete mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from model.lever_definition import LeverDefinition


@dataclass(frozen=True, slots=True)
class MechanismDefinition:
    """
    Complete mechanism definition.

    Contains all lever definitions and provides
    access helpers and consistency checks.
    """

    levers: tuple[LeverDefinition, ...]

    def __post_init__(self) -> None:
        self._validate_unique_ids()

    @property
    def lever_count(self) -> int:
        """
        Number of defined levers.
        """

        return len(self.levers)

    @property
    def input_lever(self) -> LeverDefinition:
        """
        First lever of the mechanism.

        The first lever is the driving input.
        """

        if not self.levers:
            raise ValueError("Mechanism contains no levers.")

        return self.levers[0]

    @property
    def coupled_levers(self) -> tuple[LeverDefinition, ...]:
        """
        All levers with fixed angular coupling.
        """

        return tuple(
            lever
            for lever in self.levers
            if lever.is_coupled
        )

    @property
    def driven_levers(self) -> tuple[LeverDefinition, ...]:
        """
        All levers connected through a driver relationship.
        """

        return tuple(
            lever
            for lever in self.levers
            if lever.is_driver
        )

    def get_lever(self, lever_id: int) -> LeverDefinition:
        """
        Return lever by its identifier.
        """

        for lever in self.levers:
            if lever.id == lever_id:
                return lever

        raise KeyError(f"Unknown lever id: {lever_id}")

    def _validate_unique_ids(self) -> None:
        """
        Ensure all lever IDs are unique.
        """

        ids = [lever.id for lever in self.levers]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Lever IDs must be unique."
            )