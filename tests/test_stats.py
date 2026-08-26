import math

import pytest

from prompt_experiments.stats import (
    mann_whitney,
    mde_at_n,
    required_n_proportions,
    two_proportion_test,
    welch_t,
    wilson_interval,
)


def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    """Where Wald breaks. At p̂=1 Wald reports zero width — perfect certainty from
    the least informative data possible — and at small p̂ it runs below zero."""
    perfect = wilson_interval(10, 10)
    assert perfect.high <= 1.0
    assert perfect.low < 1.0 and perfect.high - perfect.low > 0.05

    none = wilson_interval(0, 10)
    assert none.low >= 0.0 and none.high > 0.0


def test_wilson_narrows_with_more_data():
    small = wilson_interval(35, 50)
    large = wilson_interval(3500, 5000)
    assert (large.high - large.low) < (small.high - small.low) / 5


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_two_proportion_test_finds_a_real_difference():
    result = two_proportion_test(700, 1000, 780, 1000)
    assert result.p_value < 0.001
    assert result.effect == pytest.approx(0.08, abs=1e-9)
    assert result.effect_low < 0.08 < result.effect_high


def test_two_proportion_test_reports_no_difference_when_there_is_none():
    result = two_proportion_test(700, 1000, 702, 1000)
    assert result.p_value > 0.5
    assert result.effect_low < 0 < result.effect_high


def test_welch_handles_unequal_variance():
    """Student's t assumes equal variance; a chain-of-thought variant is both slower
    and far more variable than a zero-shot one, which is exactly when it misbehaves."""
    tight = [10.0 + (i % 3) * 0.1 for i in range(60)]
    wide = [10.0 + (i % 40) * 1.5 for i in range(60)]
    result = welch_t(tight, wide)
    assert result.test == "welch-t"
    assert result.n_a == 60 and result.n_b == 60
    assert math.isfinite(result.effect_low) and math.isfinite(result.effect_high)


def test_welch_is_stable_on_tiny_samples():
    result = welch_t([1.0], [2.0])
    assert result.p_value == 1.0


def test_mann_whitney_reports_medians_not_means():
    """For right-skewed latency, the median is the number a user experiences."""
    base = [100.0] * 50 + [5000.0] * 5          # long tail
    shifted = [130.0] * 50 + [5000.0] * 5
    result = mann_whitney(base, shifted)
    assert result.test == "mann-whitney"
    assert result.effect == pytest.approx(30.0, abs=1.0)


def test_required_sample_size_matches_the_textbook():
    # 70% vs 75%, alpha .05, power .8 — standard tables give roughly 1,250 per arm.
    assert required_n_proportions(0.70, 0.05) == pytest.approx(1251, abs=30)


def test_smaller_effects_need_dramatically_more_data():
    five = required_n_proportions(0.70, 0.05)
    two = required_n_proportions(0.70, 0.02)
    assert two > five * 5


def test_mde_inverts_the_sample_size_calculation():
    n = required_n_proportions(0.70, 0.05)
    assert mde_at_n(0.70, n) == pytest.approx(0.05, abs=0.005)


def test_power_inputs_are_validated():
    with pytest.raises(ValueError):
        required_n_proportions(0.0, 0.05)
    with pytest.raises(ValueError):
        required_n_proportions(0.70, 0.0)
    with pytest.raises(ValueError):
        required_n_proportions(0.98, 0.05)   # baseline + mde leaves [0,1]
