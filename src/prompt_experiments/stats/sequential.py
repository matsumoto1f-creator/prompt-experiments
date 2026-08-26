"""When it is legitimate to look at a running experiment.

**This is the module the rest of the platform exists to protect.**

The spec this project was built from says the dashboard should "display whether the
experiment has reached significance" in real time. Combined with a human who wants a
winner, that produces the oldest error in experimentation: check after every batch,
stop the moment p dips below 0.05, and report a 5% false-positive rate you are not
getting. With ten looks the real rate is closer to 20% — you are not testing a
hypothesis, you are sampling until the noise agrees with you.

`naive_peeking_error_rate()` measures exactly that, and a test asserts it. The fix is
not discipline; it is a boundary that already accounts for the looking.

**How the correction works.** Track the test statistic on the Brownian scale,
`B(t) = z(t)·√t`, where `t` is the information fraction (observations so far ÷
observations planned). Under the null, `B` is a standard Brownian motion. The
O'Brien–Fleming boundary `z_k ≥ c/√t_k` is therefore just a *constant* boundary
`|B(t_k)| ≥ c` — which is what makes it both easy to reason about and easy to
calibrate exactly: find the `c` whose crossing probability across the planned look
times is α.

That calibration is done by simulation rather than a table, so the boundary is
correct for whatever look schedule an experiment actually uses, and so the claim is
checkable — `test_sequential.py` verifies the empirical type-I error lands at α.

The boundary is conservative early on purpose: stopping at 10% of the planned sample
requires overwhelming evidence, because at that point you have almost none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

DEFAULT_DRAWS = 200_000
_CACHE: dict[tuple, float] = {}


def _brownian_maxima(look_fractions: tuple[float, ...], draws: int, seed: int) -> np.ndarray:
    """max_k |B(t_k)| for each simulated path under the null."""
    fractions = np.asarray(look_fractions, dtype=float)
    steps = np.diff(np.concatenate(([0.0], fractions)))
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, np.sqrt(steps), size=(draws, len(fractions)))
    paths = np.cumsum(increments, axis=1)
    return np.abs(paths).max(axis=1)


def obf_critical_value(
    alpha: float = 0.05,
    look_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260826,
) -> float:
    """The constant `c` such that crossing |B(t_k)| ≥ c at any planned look has
    probability α under the null. Cached per (alpha, schedule)."""
    if not look_fractions or any(t <= 0 or t > 1 for t in look_fractions):
        raise ValueError("look fractions must lie in (0, 1]")
    if list(look_fractions) != sorted(look_fractions):
        raise ValueError("look fractions must be increasing")

    key = (round(alpha, 6), tuple(round(t, 6) for t in look_fractions), draws, seed)
    if key not in _CACHE:
        maxima = _brownian_maxima(look_fractions, draws, seed)
        _CACHE[key] = float(np.quantile(maxima, 1 - alpha))
    return _CACHE[key]


def naive_peeking_error_rate(
    alpha: float = 0.05,
    look_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260826,
) -> float:
    """The false-positive rate you actually get by testing at the fixed level α at
    every look — the thing a live-significance dashboard invites.

    Two identical variants, no real difference, and this is how often a naive reader
    declares a winner anyway.
    """
    fractions = np.asarray(look_fractions, dtype=float)
    steps = np.diff(np.concatenate(([0.0], fractions)))
    rng = np.random.default_rng(seed)
    paths = np.cumsum(rng.normal(0.0, np.sqrt(steps), size=(draws, len(fractions))), axis=1)
    z_scores = np.abs(paths) / np.sqrt(fractions)
    threshold = stats.norm.ppf(1 - alpha / 2)
    return float((z_scores >= threshold).any(axis=1).mean())


@dataclass
class LookVerdict:
    look: int
    information_fraction: float
    observed_n: int
    planned_n: int
    z: float
    boundary_z: float
    crossed: bool
    reason: str


@dataclass
class SequentialPlan:
    """A pre-registered look schedule. Fix this before the experiment starts.

    Registering the schedule up front is the whole mechanism. A boundary calibrated
    for four looks does not protect an experiment someone checked forty times, so the
    plan is stored with the experiment and the analysis refuses to answer at a look
    the plan does not contain.
    """

    planned_n: int
    alpha: float = 0.05
    look_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    _critical: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.planned_n <= 0:
            raise ValueError("planned_n must be positive")
        self._critical = obf_critical_value(self.alpha, tuple(self.look_fractions))

    @property
    def critical_value(self) -> float:
        return self._critical

    def look_at(self, observed_n: int) -> int | None:
        """Which planned look this sample size corresponds to, or None if the
        experiment has not reached the next one yet."""
        fraction = observed_n / self.planned_n
        reached = [i for i, t in enumerate(self.look_fractions) if fraction >= t]
        return reached[-1] if reached else None

    def boundary_z(self, information_fraction: float) -> float:
        """The z-value a look at this information fraction must exceed."""
        return self._critical / max(information_fraction, 1e-9) ** 0.5

    def evaluate(self, z: float, observed_n: int) -> LookVerdict:
        index = self.look_at(observed_n)
        if index is None:
            first = self.look_fractions[0]
            return LookVerdict(
                look=0,
                information_fraction=observed_n / self.planned_n,
                observed_n=observed_n,
                planned_n=self.planned_n,
                z=z,
                boundary_z=float("inf"),
                crossed=False,
                reason=(
                    f"below the first planned look ({first:.0%} of {self.planned_n} = "
                    f"{int(first * self.planned_n)} observations). No verdict is available yet, "
                    "and reading the p-value here is the error this plan exists to prevent."
                ),
            )

        fraction = self.look_fractions[index]
        boundary = self.boundary_z(fraction)
        crossed = abs(z) >= boundary

        if crossed:
            reason = (
                f"look {index + 1} of {len(self.look_fractions)} at {fraction:.0%} information: "
                f"|z| = {abs(z):.2f} crossed the boundary {boundary:.2f}. "
                "Significant after accounting for every planned look."
            )
        elif index == len(self.look_fractions) - 1:
            reason = (
                f"final look: |z| = {abs(z):.2f} did not reach {boundary:.2f}. "
                "No winner — and that is a result, not a reason to keep collecting."
            )
        else:
            reason = (
                f"look {index + 1} of {len(self.look_fractions)} at {fraction:.0%} information: "
                f"|z| = {abs(z):.2f}, boundary {boundary:.2f}. Continue."
            )

        return LookVerdict(
            look=index + 1,
            information_fraction=fraction,
            observed_n=observed_n,
            planned_n=self.planned_n,
            z=z,
            boundary_z=boundary,
            crossed=crossed,
            reason=reason,
        )
