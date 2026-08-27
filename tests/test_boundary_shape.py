"""Pin the SHAPE of the sequential boundary, not just the alpha it produces.

Mutation testing killed 0 of 10 mutants in this module. The most striking:
changing the boundary from t^-0.5 to t^-2.0 — a completely different alpha
spending function — left all 44 tests green.

The reason is structural, and it is the most instructive failure in this
codebase. `_critical` is calibrated by simulation to make the family-wise error
come out at alpha. Change the exponent and the calibration simply re-solves for
a different `_critical`; the overall alpha is still 5%. Every test that checks
alpha therefore passes under any exponent, because alpha is the quantity the
calibration is DEFINED to produce. The suite was measuring the calibration's
guarantee rather than the design it was supposed to constrain.

What distinguishes O'Brien-Fleming is not its alpha — every valid spending
function hits alpha. It is that the boundary is a constant on the Brownian
scale: B(t) = z(t)*sqrt(t) does not depend on t. That is the property to pin.
"""
import pytest

from prompt_experiments.stats.sequential import SequentialPlan


@pytest.fixture
def plan():
    return SequentialPlan(planned_n=1000, alpha=0.05,
                          look_fractions=(0.25, 0.5, 0.75, 1.0))


def test_the_boundary_is_constant_on_the_brownian_scale(plan):
    """z(t) * sqrt(t) is flat. This is what makes it O'Brien-Fleming rather than
    Pocock, a power boundary, or an arbitrary curve that happens to hit alpha."""
    brownian = [plan.boundary_z(t) * t ** 0.5 for t in (0.1, 0.25, 0.5, 0.75, 1.0)]
    assert max(brownian) - min(brownian) < 1e-9, (
        f"not constant on the Brownian scale: {brownian}. The exponent in "
        "boundary_z is not -1/2."
    )
    assert brownian[0] == pytest.approx(plan._critical, abs=1e-9)


def test_the_boundary_is_strictly_conservative_early(plan):
    """The whole point of the design: an early look must clear a much higher bar
    than the final one, because stopping early on noise is the failure mode."""
    z = [plan.boundary_z(t) for t in (0.25, 0.5, 0.75, 1.0)]
    assert z == sorted(z, reverse=True), f"boundary not decreasing in information: {z}"
    # Exactly 2x, not merely "more": boundary(t) = c/sqrt(t), so at a quarter of
    # the information the bar is 1/sqrt(0.25) = 2 times the final one. Asserting
    # the exact ratio pins the exponent; asserting ">" would not.
    assert z[0] == pytest.approx(2 * z[-1], rel=1e-12)
    assert plan.boundary_z(1 / 9) == pytest.approx(3 * z[-1], rel=1e-12)
    # And the final boundary is close to, but above, the naive fixed-sample 1.96.
    assert 1.96 < z[-1] < 2.3


def test_the_final_boundary_exceeds_the_naive_critical_value(plan):
    """If the last look used 1.96 the plan would not be correcting for anything."""
    assert plan.boundary_z(1.0) > 1.96


def test_more_looks_cost_more_at_the_end(plan):
    """Peeking is not free: the price of extra looks is a stricter final boundary."""
    few = SequentialPlan(planned_n=1000, alpha=0.05, look_fractions=(0.5, 1.0))
    many = SequentialPlan(planned_n=1000, alpha=0.05,
                          look_fractions=(0.2, 0.4, 0.6, 0.8, 1.0))
    assert many.boundary_z(1.0) > few.boundary_z(1.0), (
        f"five looks ({many.boundary_z(1.0):.3f}) must end stricter than two "
        f"({few.boundary_z(1.0):.3f})"
    )


def test_a_tighter_alpha_raises_every_boundary(plan):
    strict = SequentialPlan(planned_n=1000, alpha=0.01,
                            look_fractions=(0.25, 0.5, 0.75, 1.0))
    for t in (0.25, 0.5, 0.75, 1.0):
        assert strict.boundary_z(t) > plan.boundary_z(t)


def test_crossing_is_inclusive_at_the_boundary(plan):
    """`abs(z) >= boundary`. A strict `>` survived mutation because no test ever
    sits exactly on the line — so sit on it."""
    b = plan.boundary_z(0.5)
    assert plan.evaluate(b, 500).crossed is True
    assert plan.evaluate(b - 1e-9, 500).crossed is False
    assert plan.evaluate(-b, 500).crossed is True, "the boundary is two-sided"
