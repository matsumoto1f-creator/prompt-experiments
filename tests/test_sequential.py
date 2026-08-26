"""The claims this project is built on, asserted rather than described.

If these pass, the platform's central promise holds: looking at a running experiment
on the registered schedule does not inflate the false-positive rate. If they fail,
every other feature is decoration.
"""

import pytest

from prompt_experiments.stats import (
    SequentialPlan,
    naive_peeking_error_rate,
    obf_critical_value,
)

DRAWS = 120_000


def looks(k: int) -> tuple[float, ...]:
    return tuple((i + 1) / k for i in range(k))


def test_naive_peeking_inflates_the_false_positive_rate():
    """The problem. Two identical variants, tested at alpha after every batch."""
    one = naive_peeking_error_rate(0.05, looks(1), draws=DRAWS)
    ten = naive_peeking_error_rate(0.05, looks(10), draws=DRAWS)

    assert one == pytest.approx(0.05, abs=0.006)   # a single look is honest
    assert ten > 0.15                              # ten looks is not
    assert ten > one * 3


def test_peeking_damage_grows_with_the_number_of_looks():
    rates = [naive_peeking_error_rate(0.05, looks(k), draws=40_000) for k in (1, 2, 4, 10)]
    assert rates == sorted(rates)


def test_obf_boundary_holds_the_nominal_rate():
    """The fix. Crossing the calibrated boundary at ANY planned look has probability
    alpha in total — which is the property naive testing does not have."""
    import numpy as np
    from prompt_experiments.stats.sequential import _brownian_maxima

    for k in (2, 4, 10):
        fractions = looks(k)
        c = obf_critical_value(0.05, fractions, draws=DRAWS)
        # Fresh paths on a different seed: calibrating and checking on the same draws
        # would only prove the quantile function works.
        maxima = _brownian_maxima(fractions, draws=DRAWS, seed=99)
        empirical = float((maxima >= c).mean())
        assert empirical == pytest.approx(0.05, abs=0.006), f"{k} looks gave {empirical:.3f}"


def test_obf_constant_matches_published_values():
    """O'Brien-Fleming for 4 looks at alpha=0.05 is ~2.024 in the standard tables.
    Simulation-derived rather than tabulated, so this guards the calibration itself."""
    c = obf_critical_value(0.05, (0.25, 0.5, 0.75, 1.0), draws=DRAWS)
    assert c == pytest.approx(2.024, abs=0.03)


def test_boundary_is_strict_early_and_relaxes():
    plan = SequentialPlan(planned_n=1000)
    boundaries = [plan.boundary_z(t) for t in plan.look_fractions]
    assert boundaries == sorted(boundaries, reverse=True)
    assert boundaries[0] > 3.5      # 25% information demands overwhelming evidence
    assert boundaries[-1] < 2.2     # the final look is close to the fixed-horizon value


def test_a_strongly_significant_early_look_still_does_not_stop():
    """The behaviour that separates this from a live-p-value dashboard: z=3.4 is
    p<0.001, and at 25% information it is still not enough."""
    plan = SequentialPlan(planned_n=1000)
    verdict = plan.evaluate(z=3.4, observed_n=250)
    assert not verdict.crossed
    assert verdict.look == 1


def test_no_verdict_before_the_first_planned_look():
    plan = SequentialPlan(planned_n=1000)
    verdict = plan.evaluate(z=9.9, observed_n=10)
    assert not verdict.crossed and verdict.look == 0
    assert "no verdict is available yet" in verdict.reason.lower()


def test_look_schedule_must_be_valid():
    with pytest.raises(ValueError):
        obf_critical_value(0.05, (0.5, 0.25))       # not increasing
    with pytest.raises(ValueError):
        obf_critical_value(0.05, (0.5, 1.5))        # outside (0, 1]
    with pytest.raises(ValueError):
        SequentialPlan(planned_n=0)
