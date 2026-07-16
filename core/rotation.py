"""
core/rotation.py

High level rotation utilities.

This module provides engineering-oriented rotation operations
based on Quaternion.

Quaternion contains the mathematical representation.
This module contains common operations used in kinematics.

References
----------
Jack B. Kuipers:
    Quaternions and Rotation Sequences.
    Princeton University Press, 1999.
"""

from __future__ import annotations

import math

from core.quaternion import Quaternion
from core.vector3d import Vector3D


class Rotation:
    """
    Collection of static rotation utilities.
    """

    @staticmethod
    def rotate_vector(
        vector: Vector3D,
        axis: Vector3D,
        angle_rad: float,
    ) -> Vector3D:
        """
        Rotate vector around axis by angle.

        Parameters
        ----------
        vector:
            Vector to rotate.

        axis:
            Rotation axis.

        angle_rad:
            Rotation angle in radians.
        """

        quaternion = Quaternion.from_axis_angle(
            axis,
            angle_rad,
        )

        return quaternion.rotate_vector(vector)

    @staticmethod
    def from_two_vectors(
        source: Vector3D,
        target: Vector3D,
    ) -> Quaternion:
        """
        Create quaternion rotating source vector into target vector.

        The shortest possible rotation is selected.

        Mathematical background
        -----------------------
        Given two normalized vectors:

            a = source / norm(source)
            b = target / norm(target)

        Rotation axis:

            axis = cross(a,b)

        Rotation angle:

            angle = acos(dot(a,b))

        Special cases:
            parallel vectors -> identity rotation
            opposite vectors -> 180 degree rotation
        """

        a = source.normalized()
        b = target.normalized()

        dot = a.dot(b)

        # Parallel vectors
        if dot > 1.0 - 1.0e-12:
            return Quaternion.identity()

        # Opposite vectors
        if dot < -1.0 + 1.0e-12:

            # Find a stable perpendicular axis
            if abs(a.x) < abs(a.y):
                axis = Vector3D(
                    0,
                    -a.z,
                    a.y,
                )
            else:
                axis = Vector3D(
                    -a.z,
                    0,
                    a.x,
                )

            return Quaternion.from_axis_angle(
                axis.normalized(),
                math.pi,
            )

        axis = a.cross(b)

        angle = math.acos(dot)

        return Quaternion.from_axis_angle(
            axis,
            angle,
        )

    @staticmethod
    def align_z_axis(
        target: Vector3D,
    ) -> Quaternion:
        """
        Create rotation which aligns global Z-axis with target vector.
        """

        return Rotation.from_two_vectors(
            Vector3D(0, 0, 1),
            target,
        )