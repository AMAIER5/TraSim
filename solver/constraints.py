"""
solver/constraints.py

Geometric constraint functions.

These functions are pure mathematical relations and contain
no solver logic.
"""

from __future__ import annotations

from math import sqrt

from core.point3d import Point3D


def distance(
    point_a: Point3D,
    point_b: Point3D,
) -> float:
    """
    Euclidean distance between two points.

    Formula
    -------
    distance =
        sqrt(
            dx² + dy² + dz²
        )
    """

    dx = (
        point_b.x
        -
        point_a.x
    )

    dy = (
        point_b.y
        -
        point_a.y
    )

    dz = (
        point_b.z
        -
        point_a.z
    )

    return sqrt(
        dx * dx
        +
        dy * dy
        +
        dz * dz
    )


def rod_length_error(
    point_a: Point3D,
    point_b: Point3D,
    rod_length: float,
) -> float:
    """
    Rod length residual.

    Positive:
        Rod is stretched.

    Negative:
        Rod is compressed.

    Zero:
        Constraint fulfilled.

    Formula
    -------
    error =
        current_length - rod_length
    """

    dx = (
        point_b.x
        -
        point_a.x
    )

    dy = (
        point_b.y
        -
        point_a.y
    )

    dz = (
        point_b.z
        -
        point_a.z
    )

    current_length = sqrt(
        dx * dx
        +
        dy * dy
        +
        dz * dz
    )

    return (
        current_length
        -
        rod_length
    )