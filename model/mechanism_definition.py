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
        self._validate_references()

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

    def _validate_references(self) -> None:
        """
        Every lever reference must point to a defined
        lever. Dangling ``driver``/``coupled``/``pivot_on``
        references appear when a lever is removed without
        cleaning up the levers that referenced it (issue
        #141: stale references silently persisted into
        the debug export), so they are rejected here.
        """

        ids = {lever.id for lever in self.levers}

        for lever in self.levers:
            for kind, reference in (
                ("driver", lever.driver),
                ("coupled", lever.coupled),
            ):
                if reference is not None and (
                    reference not in ids
                ):
                    raise ValueError(
                        f"Lever {lever.id}: {kind} "
                        f"references unknown lever "
                        f"{reference}."
                    )
            if lever.pivot_on is not None:
                target = lever.pivot_reference
                if target is not None and (
                    target.lever_id not in ids
                ):
                    raise ValueError(
                        f"Lever {lever.id}: pivot_on "
                        f"references unknown lever "
                        f"{target.lever_id}."
                    )

def coupled_angle(
    definition: MechanismDefinition,
    lever: LeverDefinition,
    partner_angle: float,
) -> float:
    """Angle of a torsion-coupled (shaft) lever when its
    partner lever sits at ``partner_angle``.

    A shaft coupling fixes the RELATIVE angle between the
    two levers: at the reference pose it is
    ``lever.angle_start - partner.angle_start``, and it
    stays the same in every build position (issue #149)."""
    partner = definition.get_lever(lever.coupled)
    return (
        partner_angle
        + lever.angle_start
        - partner.angle_start
    )


def propagate_coupled_angles(
    definition: MechanismDefinition,
    angles: dict[int, float],
    *,
    skip_ids: tuple[int, ...] = (),
) -> None:
    """Propagate torsion-coupled (shaft) lever angles in
    ``angles`` to a fixpoint (issue #149).

    A lever with a ``coupled`` reference that is not itself
    driven by a rod follows its partner: its angle is the
    partner's angle plus the constant coupling offset, so
    the relative angle between coupled levers is identical
    in the unbraked and the braked build.  Levers driven by
    a rod (their angle is solved from the rod) and the ids
    in ``skip_ids`` (the root drive lever) keep their angle.
    """
    while True:
        changed = False
        for lever in definition.levers:
            if (
                lever.coupled is None
                or lever.driver is not None
                or lever.id in skip_ids
            ):
                continue
            expected = coupled_angle(
                definition,
                lever,
                angles[lever.coupled],
            )
            if angles.get(lever.id) != expected:
                angles[lever.id] = expected
                changed = True
        if not changed:
            return
