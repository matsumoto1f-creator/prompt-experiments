"""Reading a running experiment.

Three independent questions, answered separately because they have different answers
and different consequences:

  guardrail     is a variant actively broken (errors), regardless of the metric?
                Stops immediately. Not a statistical question.
  sequential    are we allowed to look yet, and does the evidence clear the boundary
                for THIS look? Declares a winner.
  descriptive   what do the arms look like right now — rates, intervals, cost,
                latency? Always shown, never a basis for stopping.

Keeping the third separate from the second is the point. A dashboard that shows
current numbers is useful; a dashboard that shows a live p-value next to a Stop button
is a false-positive generator with a nice font.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_experiments.models import ArmSummary, Experiment
from prompt_experiments.stats import (
    SequentialPlan,
    mann_whitney,
    required_n_proportions,
    two_proportion_test,
    welch_t,
    wilson_interval,
)
from prompt_experiments.stats.sequential import LookVerdict
from prompt_experiments.store import Store


@dataclass
class Analysis:
    experiment: Experiment
    control: ArmSummary
    treatment: ArmSummary
    control_interval: str
    treatment_interval: str
    effect: str
    p_value: float
    test_name: str
    verdict: LookVerdict
    guardrail_breached: bool
    guardrail_detail: str
    recommendation: str

    @property
    def total_n(self) -> int:
        return self.control.n + self.treatment.n


def analyse(store: Store, experiment: Experiment) -> Analysis:
    control = store.arm(experiment.id, experiment.control_version)
    treatment = store.arm(experiment.id, experiment.treatment_version)

    breached, guardrail_detail = _check_guardrail(experiment, control, treatment)

    if experiment.metric.kind == "binary":
        result = two_proportion_test(
            control.successes, max(control.n - control.errors, 0),
            treatment.successes, max(treatment.n - treatment.errors, 0),
        )
        z, p_value, effect, test_name = result.statistic, result.p_value, result.effect_interval, "two-proportion z"
        control_interval = str(wilson_interval(control.successes, max(control.n - control.errors, 1)))
        treatment_interval = str(wilson_interval(treatment.successes, max(treatment.n - treatment.errors, 1)))
    else:
        a = store.values(experiment.id, experiment.control_version)
        b = store.values(experiment.id, experiment.treatment_version)
        result = welch_t(a, b)
        if _looks_skewed(a) or _looks_skewed(b):
            result = mann_whitney(a, b)
        z, p_value, effect, test_name = result.statistic, result.p_value, result.effect_interval, result.test
        control_interval = f"{result.mean_a:.3f}"
        treatment_interval = f"{result.mean_b:.3f}"

    plan = SequentialPlan(
        planned_n=experiment.planned_n,
        alpha=experiment.alpha,
        look_fractions=tuple(experiment.look_fractions),
    )
    # Information is bounded by the SMALLER arm: an unbalanced split does not buy
    # information at the rate the larger arm suggests.
    effective_n = min(control.n, treatment.n)
    verdict = plan.evaluate(z, effective_n)

    return Analysis(
        experiment=experiment,
        control=control,
        treatment=treatment,
        control_interval=control_interval,
        treatment_interval=treatment_interval,
        effect=effect,
        p_value=p_value,
        test_name=test_name,
        verdict=verdict,
        guardrail_breached=breached,
        guardrail_detail=guardrail_detail,
        recommendation=_recommend(experiment, control, treatment, verdict, breached, guardrail_detail),
    )


def _check_guardrail(
    experiment: Experiment, control: ArmSummary, treatment: ArmSummary
) -> tuple[bool, str]:
    rail = experiment.guardrail
    for arm, label in ((control, "control"), (treatment, "treatment")):
        if arm.n < rail.min_observations_before_check:
            continue
        if arm.error_rate > rail.max_error_rate:
            return True, (
                f"{label} arm (v{arm.version}) error rate {arm.error_rate:.1%} exceeds "
                f"the {rail.max_error_rate:.0%} guardrail over {arm.n} requests"
            )
    return False, "error rates within guardrail"


def _looks_skewed(values: list[float]) -> bool:
    """Crude skew check to pick the rank test for latency-shaped data. Deliberately
    crude — the decision only needs to be defensible, and a formal normality test on
    a few thousand points rejects for deviations too small to matter."""
    if len(values) < 20:
        return False
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    mean = sum(ordered) / len(ordered)
    spread = ordered[-1] - ordered[0]
    return spread > 0 and abs(mean - mid) / spread > 0.15


def _recommend(
    experiment: Experiment,
    control: ArmSummary,
    treatment: ArmSummary,
    verdict: LookVerdict,
    breached: bool,
    guardrail_detail: str,
) -> str:
    if breached:
        return f"STOP NOW — {guardrail_detail}. Roll back to the control version and investigate."

    if verdict.crossed:
        better = experiment.treatment_version if treatment.mean > control.mean else experiment.control_version
        if not experiment.metric.higher_is_better:
            better = experiment.control_version if treatment.mean > control.mean else experiment.treatment_version
        return (
            f"Winner: v{better}. {verdict.reason} "
            "Promote it, then hold for 24 hours before retiring the loser — a data-quality "
            "problem discovered after promotion is much cheaper to undo while the other arm still exists."
        )

    if verdict.look and verdict.look >= len(experiment.look_fractions):
        needed = required_n_proportions(experiment.baseline_estimate, experiment.mde, experiment.alpha)
        return (
            "No winner at the planned sample size. That is a finding: the difference, if any, "
            f"is smaller than the {experiment.mde:.0%} this experiment was powered to detect "
            f"(which needed {needed} per arm). Ship the simpler variant, or re-run powered for a "
            "smaller effect if one that size is worth the traffic."
        )

    remaining = max(0, experiment.planned_n - min(control.n, treatment.n))
    return f"Keep running — {remaining} more observations per arm to the planned sample size. {verdict.reason}"
