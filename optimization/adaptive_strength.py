"""
optimization/adaptive_strength.py

Global adaptive step-size control via the Rechenberg
1/5 success rule.

The 1/5 success rule (Rechenberg, 1973) is the classic
self-controlling mechanism for the mutation strength of
evolution strategies:

- Over a sliding window of generations, count how often
  a new best solution was found (a "success").
- If the success rate is above the target rate (1/5),
  the step size is *increased*: the search is progressing
  well, so larger steps explore further.
- If the success rate is below the target rate, the step
  size is *decreased*: too many failed steps, so smaller
  steps exploit the current region.

This keeps the mutation strength balanced automatically
instead of relying on a fixed, manually tuned value.

The controller operates on the mutation operator passed
to it: after each recorded outcome it updates
``mutation.strength`` in place, clamped to a configurable
range so the step size never collapses to zero or grows
without bound.
"""

from __future__ import annotations

from collections import deque

from optimization.parameter_mutation import (
    ParameterMutation,
)

# Rechenberg's classic target success rate.
DEFAULT_TARGET_SUCCESS_RATE = 1.0 / 5.0

# Multiplicative update factors.  A factor > 1 increases
# the step size on success, a factor < 1 decreases it on
# failure.  Using the same factor in both directions keeps
# the rule symmetric and unbiased.
DEFAULT_UPDATE_FACTOR = 1.2

# Sensible bounds for the relative step size (fraction of
# the parameter range).  Too small and the search stalls;
# too large and the search becomes random.
DEFAULT_MIN_STRENGTH = 1e-4
DEFAULT_MAX_STRENGTH = 1.0


class AdaptiveStrength:
    """
    Adaptive step-size controller using the 1/5 success
    rule.

    The controller is driven by :meth:`record`, which is
    called once per generation with whether the best
    fitness improved.  Every ``window`` generations the
    success rate is evaluated and the controlled
    mutation's ``strength`` is updated in place.
    """

    def __init__(
        self,
        mutation: ParameterMutation,
        *,
        window: int = 10,
        target_success_rate: float = DEFAULT_TARGET_SUCCESS_RATE,
        update_factor: float = DEFAULT_UPDATE_FACTOR,
        min_strength: float = DEFAULT_MIN_STRENGTH,
        max_strength: float = DEFAULT_MAX_STRENGTH,
    ) -> None:

        if window <= 0:

            raise ValueError(
                "window must be positive"
            )

        if not 0.0 < target_success_rate < 1.0:

            raise ValueError(
                "target_success_rate must be in (0, 1)"
            )

        if update_factor <= 1.0:

            raise ValueError(
                "update_factor must be greater than 1"
            )

        if min_strength < 0.0:

            raise ValueError(
                "min_strength must be non-negative"
            )

        if max_strength < min_strength:

            raise ValueError(
                "max_strength must be >= min_strength"
            )

        self.mutation = mutation

        self.window = window

        self.target_success_rate = target_success_rate

        self.update_factor = update_factor

        self.min_strength = min_strength

        self.max_strength = max_strength

        self._outcomes: deque[bool] = deque(
            maxlen=window,
        )

        # History of the strengths applied, exposed for
        # diagnostics (e.g. printing the adaptation curve).
        self.history: list[float] = [
            mutation.strength,
        ]

    @property
    def strength(self) -> float:
        """
        Current controlled mutation strength.
        """

        return self.mutation.strength

    @property
    def success_rate(self) -> float:
        """
        Success rate over the current window.

        Returns 0.0 when no outcomes have been recorded yet.
        """

        if not self._outcomes:

            return 0.0

        successes = sum(self._outcomes)

        return successes / len(self._outcomes)

    def record(self, improved: bool) -> None:
        """
        Record one generation's outcome and adapt the
        mutation strength when the window is full.

        Parameters
        ----------
        improved:
            True if the best fitness improved in this
            generation, False otherwise.
        """

        self._outcomes.append(improved)

        if len(self._outcomes) < self.window:

            return

        self._adapt()

    def _adapt(self) -> None:
        """
        Apply the 1/5 success rule to the controlled
        mutation strength.
        """

        rate = self.success_rate

        if rate > self.target_success_rate:

            # More successes than the target: take larger
            # steps to explore further.
            factor = self.update_factor

        elif rate < self.target_success_rate:

            # Fewer successes than the target: take smaller
            # steps to exploit the current region.
            factor = 1.0 / self.update_factor

        else:

            return

        new_strength = self.mutation.strength * factor

        new_strength = max(
            self.min_strength,
            min(
                self.max_strength,
                new_strength,
            ),
        )

        self.mutation.strength = new_strength

        self.history.append(new_strength)

        # Reset the window so the next adaptation uses a
        # fresh, non-overlapping sample.
        self._outcomes.clear()
