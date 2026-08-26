"""Binary outcomes — the most common prompt metric.

"Did the response clear the quality bar" is a proportion, and proportions at the
sample sizes prompt experiments actually reach (hundreds, not millions) are where the
textbook normal approximation quietly misbehaves.

Wilson intervals rather than Wald (`p̂ ± z·√(p̂(1-p̂)/n)`), because Wald is wrong in
exactly the situations that matter here: it produces intervals that run below 0 or
above 1, and at p̂ = 0 or 1 it collapses to zero width — reporting perfect certainty
from the least informative data you can have. Wilson stays inside [0, 1] and keeps
sensible width at the boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    effect: float          # difference in proportions, arm B minus arm A
    effect_low: float
    effect_high: float
    n_a: int
    n_b: int

    @property
    def effect_interval(self) -> str:
        return f"{self.effect:+.1%} [{self.effect_low:+.1%}, {self.effect_high:+.1%}]"


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a single proportion."""
    if n <= 0:
        return Interval(0.0, 0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin))


def two_proportion_test(
    successes_a: int, n_a: int, successes_b: int, n_b: int, confidence: float = 0.95
) -> TestResult:
    """Two-sided test of H0: p_a == p_b, with an interval on the difference.

    Note the two variances: the test statistic pools under the null (that is what the
    null asserts), while the confidence interval on the difference does not (under the
    alternative the proportions differ, so pooling would understate the spread). Using
    one for both is a common and quiet error — it makes the interval disagree with the
    p-value near the boundary.
    """
    if n_a <= 0 or n_b <= 0:
        return TestResult(0.0, 1.0, 0.0, -1.0, 1.0, n_a, n_b)

    p_a, p_b = successes_a / n_a, successes_b / n_b
    effect = p_b - p_a

    pooled = (successes_a + successes_b) / (n_a + n_b)
    se_null = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = effect / se_null if se_null > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    return TestResult(
        statistic=z,
        p_value=p_value,
        effect=effect,
        effect_low=effect - crit * se_diff,
        effect_high=effect + crit * se_diff,
        n_a=n_a,
        n_b=n_b,
    )
