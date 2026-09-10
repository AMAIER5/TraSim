"""
optimization/parameter_mutation.py

Mutation operator for optimization parameters.

The mutation creates modified parameter sets.

Distributions
-------------
By default each parameter is perturbed by a value drawn
from a *Gaussian* distribution centred on the current
value.  A Gaussian distribution produces many small
steps (fine local tuning) and few large steps
(exploration), which is the standard behaviour for
evolution strategies.  The legacy uniform box
distribution is still available via
``distribution="uniform"``.

Strength / step size
--------------------
``strength`` is the relative step size as a fraction of
the parameter range (``maximum - minimum``).  A Gaussian
mutation draws

    delta = N(0, 1) * range * strength

so that ~68 % of all steps stay within
``+/- strength * range`` of the parent value.

Boundary handling
-----------------
Mutated values are kept inside the parameter range by
*reflecting* them at the boundaries instead of clamping.
Reflection avoids the wall effect (candidates piling up
at the range limits) and preserves the distribution
shape near the boundaries.
"""

from __future__ import annotations

import random

from optimization.parameter import (
    Parameter,
)


def _reflect(value: float, minimum: float, maximum: float) -> float:
    """
    Reflect a value back into [minimum, maximum].

    Mirroring at the boundaries preserves the distribution
    shape and avoids the wall effect that clamping produces.
    The reflection is computed in closed form via a
    triangular wave so that arbitrarily large deltas are
    folded into the interval in O(1).

    Degenerate ranges (minimum == maximum) return the
    single valid value.
    """

    if minimum == maximum:

        return minimum

    span = maximum - minimum

    # Normalise to a zero-based interval of width span.
    offset = value - minimum

    # Map the offset into one triangular period of width
    # 2*span using the periodicity of the reflection.  The
    # modulo result is always in [0, 2*span).
    period = 2.0 * span
    folded = offset - period * int(offset // period)

    # The first half of the period maps directly, the
    # second half is mirrored back.
    if folded > span:

        folded = period - folded

    reflected = minimum + folded

    # Guard against tiny floating point overshoot.
    if reflected < minimum:

        reflected = minimum

    elif reflected > maximum:

        reflected = maximum

    return reflected
from optimization.parameter_set import (
    ParameterSet,
)

# Available perturbation distributions.
UNIFORM = "uniform"
GAUSS = "gauss"

_VALID_DISTRIBUTIONS = (UNIFORM, GAUSS)


class ParameterMutation:
    """
    Creates modified parameter sets.

    Mutation uses a relative change based
    on the parameter range.
    """

    def __init__(
        self,
        *,
        strength: float = 0.1,
        random_generator: random.Random | None = None,
        distribution: str = GAUSS,
    ) -> None:

        if strength < 0.0:

            raise ValueError(
                "strength must be positive"
            )

        if distribution not in _VALID_DISTRIBUTIONS:

            raise ValueError(
                f"distribution must be one of "
                f"{_VALID_DISTRIBUTIONS}, got {distribution!r}"
            )

        self.strength = strength

        self.random = (
            random_generator
            if random_generator is not None
            else random.Random()
        )

        self.distribution = distribution

    def apply(
        self,
        parameter_set: ParameterSet,
    ) -> ParameterSet:
        """
        Create mutated copy.
        """

        parameters = []

        for parameter in parameter_set.parameters:

            value = self._mutate_value(
                parameter
            )

            parameters.append(
                Parameter(
                    name=parameter.name,
                    minimum=parameter.minimum,
                    maximum=parameter.maximum,
                    value=value,
                )
            )

        return ParameterSet(
            tuple(parameters)
        )

    def _mutate_value(
        self,
        parameter: Parameter,
    ) -> float:
        """
        Mutate single value.
        """

        if self.strength == 0.0:

            return parameter.value

        range_size = (
            parameter.maximum
            -
            parameter.minimum
        )

        if range_size == 0.0:

            return parameter.value

        delta = self._draw_delta(
            range_size
        )

        value = (
            parameter.value
            +
            delta
        )

        return _reflect(
            value,
            parameter.minimum,
            parameter.maximum,
        )

    def _draw_delta(
        self,
        range_size: float,
    ) -> float:
        """
        Draw a perturbation delta for one parameter.
        """

        if self.distribution == UNIFORM:

            return (
                self.random.uniform(
                    -1.0,
                    1.0,
                )
                *
                range_size
                *
                self.strength
            )

        # Default: Gaussian mutation.
        return (
            self.random.gauss(
                0.0,
                1.0,
            )
            *
            range_size
            *
            self.strength
        )
