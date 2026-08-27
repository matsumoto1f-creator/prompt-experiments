"""The sequential boundary is calibrated on the STANDARD NORMAL scale.

Every test in stats/ reports a `statistic`, but they are not on the same scale:
the two-proportion test reports a z, Welch reports a t, and Mann-Whitney reports
U, which lives in [0, n_a*n_b]. Passing `result.statistic` through to a boundary
of ~2.0 therefore compares a number in the hundreds of thousands against 2.0.

These tests pin the scale, not the plumbing.
"""
import numpy as np
import pytest

from prompt_experiments.stats.continuous import mann_whitney, welch_t
from prompt_experiments.stats.proportions import two_proportion_test
from prompt_experiments.stats.sequential import SequentialPlan

PLAN = SequentialPlan(planned_n=1000, alpha=0.05, look_fractions=(0.25, 0.5, 0.75, 1.0))


def test_mann_whitney_on_identical_distributions_does_not_declare_a_winner():
    """The bug in one line: U is not a z. Two samples drawn from the SAME
    distribution must not cross a 5% boundary at the first look."""
    rng = np.random.default_rng(7)
    a = rng.lognormal(3.0, 1.0, 500)   # skewed, so analyse() routes to mann-whitney
    b = rng.lognormal(3.0, 1.0, 500)   # same parameters: no effect exists

    result = mann_whitney(a, b)
    assert result.p_value > 0.05, "sanity: these samples really are indistinguishable"

    verdict = PLAN.evaluate(result.z_score, 500)
    assert not verdict.crossed, (
        f"declared significance on identical distributions with |z|={abs(result.z_score):.2f} "
        f"against boundary {verdict.boundary_z:.2f}"
    )


def test_the_standardised_statistic_agrees_with_the_p_value_for_every_test():
    """A z-score and a two-sided p-value are two views of one number. If they
    disagree, one of them is lying about how much evidence there is."""
    from scipy import stats as sp
    rng = np.random.default_rng(11)
    cases = [
        two_proportion_test(400, 1000, 460, 1000),
        welch_t(rng.normal(0, 1, 300), rng.normal(0.3, 1, 300)),
        mann_whitney(rng.lognormal(3, 1, 300), rng.lognormal(3.4, 1, 300)),
    ]
    for result in cases:
        expected_p = 2 * sp.norm.sf(abs(result.z_score))
        assert expected_p == pytest.approx(result.p_value, abs=0.02), (
            f"{getattr(result, 'test', 'two-proportion')}: z={result.z_score:.3f} implies "
            f"p={expected_p:.4f} but the test reports p={result.p_value:.4f}"
        )


def test_the_standardised_statistic_keeps_the_direction_of_the_effect():
    """Sign carries which arm won. A boundary check on |z| hides a sign error,
    so the sign has to be asserted directly."""
    rng = np.random.default_rng(3)
    slower = mann_whitney(rng.lognormal(3.0, 0.5, 400), rng.lognormal(3.6, 0.5, 400))
    faster = mann_whitney(rng.lognormal(3.6, 0.5, 400), rng.lognormal(3.0, 0.5, 400))
    assert slower.z_score > 0 and slower.effect > 0
    assert faster.z_score < 0 and faster.effect < 0

    worse = welch_t(rng.normal(1.0, 1, 300), rng.normal(0.0, 1, 300))
    assert worse.z_score < 0 and worse.effect < 0


def test_a_real_effect_still_crosses():
    """The fix must not simply make everything non-significant — that would pass
    the test above for the wrong reason."""
    rng = np.random.default_rng(5)
    a = rng.lognormal(3.0, 0.6, 500)
    b = rng.lognormal(3.5, 0.6, 500)
    verdict = PLAN.evaluate(mann_whitney(a, b).z_score, 500)
    assert verdict.crossed
