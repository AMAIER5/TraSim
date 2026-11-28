# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
solver/_rod_residual.pyx

Cython transpilation of solver/_rod_residual.py.

Build in place with::

    python setup.py build_ext --inplace

The compiled extension module takes import precedence over the
pure-Python ``_rod_residual.py`` in the same package, so
``from solver._rod_residual import ...`` automatically uses the
native code when the extension has been built.  Without a build,
the pure-Python module is used and results are bit-identical.
"""

from libc.math cimport cos, sin, sqrt


def rod_length_residual(
    double output_x,
    double output_y,
    double output_z,
    double input_x,
    double input_y,
    double input_z,
    double rod_length,
):
    """
    Rod length residual from plain float endpoint components.
    """

    cdef double dx = output_x - input_x
    cdef double dy = output_y - input_y
    cdef double dz = output_z - input_z

    return sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    ) - rod_length


def stage_residual(
    double angle,
    double pivot_x,
    double pivot_y,
    double pivot_z,
    double axis_x,
    double axis_y,
    double axis_z,
    double reference_x,
    double reference_y,
    double reference_z,
    double cross_x,
    double cross_y,
    double cross_z,
    double dot,
    double length,
    double input_x,
    double input_y,
    double input_z,
    double rod_length,
):
    """
    Fused lever endpoint + rod length residual in one native call.
    """

    cdef double c = cos(angle)
    cdef double s = sin(angle)

    cdef double direction_x = (
        reference_x * c
        + cross_x * s
        + axis_x * dot * (1.0 - c)
    )

    cdef double direction_y = (
        reference_y * c
        + cross_y * s
        + axis_y * dot * (1.0 - c)
    )

    cdef double direction_z = (
        reference_z * c
        + cross_z * s
        + axis_z * dot * (1.0 - c)
    )

    cdef double dx = (
        pivot_x + direction_x * length
    ) - input_x

    cdef double dy = (
        pivot_y + direction_y * length
    ) - input_y

    cdef double dz = (
        pivot_z + direction_z * length
    ) - input_z

    return sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    ) - rod_length


def make_stage_residual(
    constants,
    double input_x,
    double input_y,
    double input_z,
    double rod_length,
):
    """
    Create a residual closure for one fixed input position.

    Compiled closure: the lever constants are unpacked once, so
    each evaluation is a single native call.
    """

    cdef double pivot_x = constants[0]
    cdef double pivot_y = constants[1]
    cdef double pivot_z = constants[2]
    cdef double axis_x = constants[3]
    cdef double axis_y = constants[4]
    cdef double axis_z = constants[5]
    cdef double reference_x = constants[6]
    cdef double reference_y = constants[7]
    cdef double reference_z = constants[8]
    cdef double cross_x = constants[9]
    cdef double cross_y = constants[10]
    cdef double cross_z = constants[11]
    cdef double dot = constants[12]
    cdef double length = constants[13]

    def residual(
        double angle,
    ):

        return stage_residual(
            angle,
            pivot_x,
            pivot_y,
            pivot_z,
            axis_x,
            axis_y,
            axis_z,
            reference_x,
            reference_y,
            reference_z,
            cross_x,
            cross_y,
            cross_z,
            dot,
            length,
            input_x,
            input_y,
            input_z,
            rod_length,
        )

    return residual
