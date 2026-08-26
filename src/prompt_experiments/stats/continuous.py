"""Real-valued outcomes — latency, cost, token counts, judge scores.

Welch's t-test is the default rather than Student's. Student's assumes the two arms
have equal variance, and prompt variants routinely violate that on purpose: a
chain-of-thought variant is both slower on average and far more variable than a
zero-shot one. Welch costs nothing when variances happen to be equal and is correct
when they are not, so there is no case for the other one.

Mann-Whitney is offered for distributions where the mean is not the interesting
statistic. Latency is the standard example — it is right-skewed with a long tail, and
a difference in medians is often what a user actually experiences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ContinuousResult:
    test: str
    statistic: float
    p_value: float
    mean_a: float
    mean_b: float
    effect: float
    effect_low: float
    effect_high: float
    n_a: int
    n_b: int

    @property
    def effect_interval(self) -> str:
        return f"{self.effect:+.3f} [{self.effect_low:+.3f}, {self.effect_high:+.3f}]"


def welch_t(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> ContinuousResult:
    """Welch's unequal-variance t-test, with a confidence interval on the difference."""
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if arr_a.size < 2 or arr_b.size < 2:
        return ContinuousResult("welch-t", 0.0, 1.0, float(arr_a.mean() if arr_a.size else 0),
                                float(arr_b.mean() if arr_b.size else 0), 0.0, -math.inf, math.inf,
                                arr_a.size, arr_b.size)

    statistic, p_value = stats.ttest_ind(arr_b, arr_a, equal_var=False)

    var_a, var_b = arr_a.var(ddof=1), arr_b.var(ddof=1)
    se = math.sqrt(var_a / arr_a.size + var_b / arr_b.size)
    # Welch-Satterthwaite degrees of freedom — not n_a + n_b - 2.
    df = (var_a / arr_a.size + var_b / arr_b.size) ** 2 / (
        (var_a / arr_a.size) ** 2 / (arr_a.size - 1) + (var_b / arr_b.size) ** 2 / (arr_b.size - 1)
    ) if se > 0 else 1.0
    crit = stats.t.ppf(1 - (1 - confidence) / 2, df)
    effect = float(arr_b.mean() - arr_a.mean())

    return ContinuousResult(
        test="welch-t",
        statistic=float(statistic),
        p_value=float(p_value),
        mean_a=float(arr_a.mean()),
        mean_b=float(arr_b.mean()),
        effect=effect,
        effect_low=effect - crit * se,
        effect_high=effect + crit * se,
        n_a=int(arr_a.size),
        n_b=int(arr_b.size),
    )


def mann_whitney(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> ContinuousResult:
    """Rank-based comparison, for skewed metrics where the mean is not the story.

    The reported effect is the difference in medians, and its interval is bootstrapped
    — there is no closed form, and quoting a mean-difference interval next to a
    rank-based p-value would be describing two different questions as one.
    """
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if arr_a.size < 2 or arr_b.size < 2:
        return ContinuousResult("mann-whitney", 0.0, 1.0, 0.0, 0.0, 0.0, -math.inf, math.inf,
                                arr_a.size, arr_b.size)

    statistic, p_value = stats.mannwhitneyu(arr_b, arr_a, alternative="two-sided")
    effect = float(np.median(arr_b) - np.median(arr_a))

    rng = np.random.default_rng(0)  # fixed: a reported interval must not move between runs
    draws = np.array([
        np.median(rng.choice(arr_b, arr_b.size, replace=True))
        - np.median(rng.choice(arr_a, arr_a.size, replace=True))
        for _ in range(2000)
    ])
    low, high = np.quantile(draws, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])

    return ContinuousResult(
        test="mann-whitney",
        statistic=float(statistic),
        p_value=float(p_value),
        mean_a=float(np.median(arr_a)),
        mean_b=float(np.median(arr_b)),
        effect=effect,
        effect_low=float(low),
        effect_high=float(high),
        n_a=int(arr_a.size),
        n_b=int(arr_b.size),
    )
