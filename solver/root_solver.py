"""
solver/root_solver.py

Generic numerical root finding utilities.
"""

from __future__ import annotations

import math

from collections.abc import Callable


class RootSolver:
    """
    Generic one-dimensional root finder.
    """

    @staticmethod
    def find_bracket(
        function: Callable[[float], float],
        center: float,
        window: float,
        step: float,
    ) -> tuple[float, float, int] | None:
        """
        Find the nearest interval containing a sign change.

        The search expands symmetrically around the previous
        solution to preserve the current motion branch.

        Returns
        -------
        (left, right, evaluations)

        or

        None
        """

        if window <= 0.0:
            raise ValueError("window must be positive.")

        if step <= 0.0:
            raise ValueError("step must be positive.")

        evaluations = 1

        center_value = function(center)

        if center_value == 0.0:
            return center, center, evaluations

        max_index = int(math.ceil(window / step))

        #
        # Positive direction
        #

        previous_angle = center
        previous_value = center_value

        for index in range(1, max_index + 1):

            angle = center + index * step
            value = function(angle)

            evaluations += 1

            if previous_value * value <= 0.0:
                return (
                    previous_angle,
                    angle,
                    evaluations,
                )

            previous_angle = angle
            previous_value = value

        #
        # Negative direction
        #

        previous_angle = center
        previous_value = center_value

        for index in range(1, max_index + 1):

            angle = center - index * step
            value = function(angle)

            evaluations += 1

            if previous_value * value <= 0.0:
                return (
                    angle,
                    previous_angle,
                    evaluations,
                )

            previous_angle = angle
            previous_value = value

        return None

    @staticmethod
    def solve_brent(
        function: Callable[[float], float],
        left: float,
        right: float,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 40,
    ) -> tuple[float, float, int]:
        """
        Solve a bracketed root using Brent's method.
        """

        fa = function(left)
        fb = function(right)

        iterations = 2

        if fa == 0.0:
            return left, fa, iterations

        if fb == 0.0:
            return right, fb, iterations

        if fa * fb > 0.0:
            raise ValueError(
                "Interval does not bracket a root."
            )

        a = left
        b = right
        c = a

        fc = fa

        d = b - a
        e = d
        
        while iterations < max_iterations:

            if abs(fc) < abs(fb):

                a, b, c = (
                    b,
                    c,
                    b,
                )

                fa, fb, fc = (
                    fb,
                    fc,
                    fb,
                )

            tolerance_step = (
                2.0
                * math.ulp(1.0)
                * abs(b)
                + tolerance
            )

            midpoint = 0.5 * (c - b)

            if (
                abs(midpoint) <= tolerance_step
                or fb == 0.0
            ):
                return (
                    b,
                    fb,
                    iterations,
                )

            if (
                abs(e) >= tolerance_step
                and abs(fa) > abs(fb)
            ):

                s = fb / fa

                if a == c:

                    p = (
                        2.0
                        * midpoint
                        * s
                    )

                    q = 1.0 - s

                else:

                    q = fa / fc
                    r = fb / fc

                    p = (
                        s
                        * (
                            2.0
                            * midpoint
                            * q
                            * (q - r)
                            -
                            (b - a)
                            * (r - 1.0)
                        )
                    )

                    q = (
                        (q - 1.0)
                        * (r - 1.0)
                        * (s - 1.0)
                    )

                if p > 0.0:
                    q = -q
                else:
                    p = -p

                if (
                    q != 0.0
                    and
                    2.0 * p
                    <
                    min(
                        3.0 * midpoint * q
                        - abs(tolerance * q),
                        abs(e * q),
                    )
                ):

                    e = d
                    d = p / q

                else:

                    d = midpoint
                    e = midpoint

            else:

                d = midpoint
                e = midpoint

            a = b
            fa = fb

            if abs(d) > tolerance_step:
                b += d
            else:
                b += math.copysign(
                    tolerance_step,
                    midpoint,
                )

            fb = function(b)
            iterations += 1

            if (
                (fb > 0.0 and fc > 0.0)
                or
                (fb < 0.0 and fc < 0.0)
            ):

                c = a
                fc = fa

                d = b - a
                e = d

        return (
            b,
            fb,
            iterations,
        )