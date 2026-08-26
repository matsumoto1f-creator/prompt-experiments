"""How many observations the question needs — computed before it is asked.

An experiment without a planned sample size cannot be analysed sequentially (the
boundary is defined against the plan) and cannot be stopped honestly (there is no
point at which "no difference" becomes a finding rather than an intermission).

The number is usually larger than people expect, and that is the useful part: being
told up front that detecting a 3-point improvement needs 3,500 requests per arm is
what stops a two-day experiment from being run as though it could answer the question.
"""

from __future__ import annotations

import math

from scipy import stats


def required_n_proportions(
    baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Observations per arm to detect an absolute change of `mde` from `baseline`.

    `mde` is the minimum difference worth detecting, in absolute percentage points —
    a baseline of 0.70 with mde 0.05 asks about 70% vs 75%. Absolute rather than
    relative because it is the version people state correctly out loud.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must lie strictly between 0 and 1")
    if mde <= 0:
        raise ValueError("mde must be positive")

    treated = baseline + mde
    if not 0 < treated < 1:
        raise ValueError(f"baseline {baseline} plus mde {mde} leaves the [0,1] range")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pooled = (baseline + treated) / 2

    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(baseline * (1 - baseline) + treated * (1 - treated))
    ) ** 2
    return math.ceil(numerator / (mde ** 2))


def mde_at_n(baseline: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """The smallest effect `n` per arm can detect — the inverse question, and the one
    worth asking when the sample size is fixed by traffic rather than by choice.

    Answering it prevents the most common wasted experiment: running on the traffic
    available and concluding "no difference" when the design could never have seen
    the difference that was there.
    """
    if n <= 0:
        return 1.0
    low, high = 1e-6, min(baseline, 1 - baseline) - 1e-6
    if high <= low:
        return 1.0
    for _ in range(60):  # bisection; required_n is monotone decreasing in mde
        mid = (low + high) / 2
        if required_n_proportions(baseline, mid, alpha, power) > n:
            low = mid
        else:
            high = mid
    return high
