"""
solver/root_solver.py

Generic numerical root finding utilities.

Issue #20: The Brent implementation had two concerns:

1. ``iterations`` started at 2 and loosely counted function
   evaluations rather than iterations.  It is now initialized
   to 0 (the first two evaluations are setup, not iterations)
   and incremented only inside the loop body, so it accurately
   reflects the number of *refinement steps*.

2. The inverse-quadratic-interpolation acceptance condition
   used ``abs(tolerance * q)`` (the raw user tolerance) where
   the classic Brent (Numerical Recipes ``zbrent``) uses a
   ``tol1``-style guard that combines the user tolerance with
   machine epsilon.  This is now replaced by
   ``abs(tol1 * q)`` where ``tol1`` is the same convergence
   threshold used for the midpoint check, ensuring the
   interpolation step is only accepted when it is large
   relative to both the user tolerance *and* the floating-point
   resolution at the current best estimate.
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
        Find a root bracket around a center position.

        Searches from center towards both limits:

            center - window
            center + window

        The closest sign change is returned.
        """

        if window <= 0.0:

            raise ValueError(
                "window must be positive."
            )

        if step <= 0.0:

            raise ValueError(
                "step must be positive."
            )

        evaluations = 0

        center_value = function(center)

        evaluations += 1


        if center_value == 0.0:

            return (
                center,
                center,
                evaluations,
            )


        max_steps = int(
            math.ceil(
                window / step
            )
        )


        #
        # Search positive direction
        #

        previous_angle = center
        previous_value = center_value


        for i in range(1, max_steps + 1):

            angle = center + i * step

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
        # Search negative direction
        #

        previous_angle = center
        previous_value = center_value


        for i in range(1, max_steps + 1):

            angle = center - i * step

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
    def find_all_brackets(
        function: Callable[[float], float],
        minimum: float,
        maximum: float,
        step: float,
    ) -> list[tuple[float, float, int]]:
        """
        Find all intervals containing a sign change.

        The search scans the complete interval from minimum
        to maximum with a fixed step size.

        Returns:

            (left, right, evaluations)

        for every detected root bracket.
        """

        if minimum >= maximum:

            raise ValueError(
                "minimum must be smaller than maximum."
            )

        if step <= 0.0:

            raise ValueError(
                "step must be positive."
            )


        brackets: list[
            tuple[float, float, int]
        ] = []


        evaluations = 0


        left = minimum

        left_value = function(left)

        evaluations += 1


        while left < maximum:

            right = min(
                left + step,
                maximum,
            )

            right_value = function(right)
            evaluations += 1


            #
            # Exact root at left boundary
            #

            if left_value == 0.0:

                brackets.append(
                    (
                        left,
                        left,
                        evaluations,
                    )
                )

            #
            # Sign change
            #

            elif left_value * right_value < 0.0:

                brackets.append(
                    (
                        left,
                        right,
                        evaluations,
                    )
                )

            #
            # Exact root at right boundary
            #

            if right_value == 0.0:

                brackets.append(
                    (
                        right,
                        right,
                        evaluations,
                    )
                )

            left = right

            left_value = right_value

        return brackets


    @staticmethod
    def find_all_brackets_around(
        function: Callable[[float], float],
        center: float,
        window: float,
        step: float,
    ) -> list[tuple[float, float, int]]:
        """
        Find all root brackets around a predicted position.

        Searches only within:

            center - window
            center + window

        This keeps the solver local while preserving
        multiple possible mathematical branches.
        """

        if window <= 0.0:

            raise ValueError(
                "window must be positive."
            )


        return RootSolver.find_all_brackets(
            function=function,
            minimum=center - window,
            maximum=center + window,
            step=step,
        )



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

        Issue #20: ``iterations`` now counts refinement
        steps (starting from 0), not function evaluations.
        The IQI acceptance condition now uses a ``tol1``-
        style guard instead of the raw user tolerance.
        """

        fa = function(left)

        fb = function(right)


        # Issue #20: Count iterations, not function evals.
        # The two evaluations above are setup.
        iterations = 0


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


            # Issue #20: tol1-style convergence threshold
            # combining machine epsilon and user tolerance.
            tolerance_step = (
                2.0
                * math.ulp(1.0)
                * abs(b)
                +
                tolerance
            )


            midpoint = 0.5 * (c - b)


            if (
                abs(midpoint)
                <= tolerance_step
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
                        *
                        (r - 1.0)
                        *
                        (s - 1.0)
                    )



                if p > 0.0:

                    q = -q

                else:

                    p = -p


                # Issue #20: Use tol1 (tolerance_step) instead
                # of raw tolerance in the acceptance guard,
                # matching the classic Brent (zbrent) form.
                if (
                    q != 0.0
                    and
                    2.0 * p
                    <
                    min(
                        3.0 * midpoint * q
                        -
                        abs(tolerance_step * q),
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