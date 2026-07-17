"""
solver/constraints.py

Geometric constraint functions.

These functions are pure mathematical relations and contain
no solver logic.
"""

from __future__ import annotations

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
        norm(point_b - point_a)
    """

    return (point_b - point_a).norm()


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

    return (
        distance(
            point_a,
            point_b,
        )
        - rod_length
    )