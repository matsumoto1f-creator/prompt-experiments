"""Experiment lifecycle.

The rule that does the most work here: **the plan is fixed at start and immutable
afterwards**. `planned_n`, `look_fractions` and `alpha` are what the sequential
boundary is calibrated against, so editing them mid-flight does not adjust the test —
it silently invalidates it while leaving every number on screen looking fine.

Attempting it raises. That is the one piece of rigidity worth having.
"""

from __future__ import annotations

from datetime import datetime, timezone

from prompt_experiments.analysis import analyse
from prompt_experiments.models import (
    AuditEntry,
    Experiment,
    Guardrail,
    MetricSpec,
    StopReason,
)
from prompt_experiments.registry import set_active
from prompt_experiments.stats import required_n_proportions
from prompt_experiments.store import Store

FROZEN_FIELDS = ("planned_n", "look_fractions", "alpha", "control_version",
                 "treatment_version", "traffic_split", "metric")


class ExperimentError(ValueError):
    pass


def create_experiment(
    store: Store,
    experiment_id: str,
    prompt_id: str,
    control_version: int,
    treatment_version: int,
    *,
    metric: MetricSpec | None = None,
    baseline: float = 0.7,
    mde: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.8,
    traffic_split: float = 0.5,
    planned_n: int | None = None,
    look_fractions: list[float] | None = None,
    guardrail: Guardrail | None = None,
) -> Experiment:
    if store.get_experiment(experiment_id):
        raise ExperimentError(f"experiment {experiment_id!r} already exists")
    if control_version == treatment_version:
        raise ExperimentError("control and treatment must be different versions")
    for version in (control_version, treatment_version):
        if not store.get_version(prompt_id, version):
            raise ExperimentError(f"no v{version} of prompt {prompt_id!r}")

    # Derived rather than guessed. Being told up front that a 2-point improvement
    # needs 8,000 per arm is the number that stops a two-day experiment being run as
    # though it could answer the question.
    computed_n = planned_n or required_n_proportions(baseline, mde, alpha, power)

    experiment = Experiment(
        id=experiment_id,
        prompt_id=prompt_id,
        control_version=control_version,
        treatment_version=treatment_version,
        metric=metric or MetricSpec(name="quality", kind="binary"),
        traffic_split=traffic_split,
        planned_n=computed_n,
        look_fractions=look_fractions or [0.25, 0.5, 0.75, 1.0],
        alpha=alpha,
        baseline_estimate=baseline,
        mde=mde,
        guardrail=guardrail or Guardrail(),
    )
    store.save_experiment(experiment)
    store.audit(AuditEntry(
        actor="system", action="create_experiment", subject=experiment_id,
        reason=f"v{control_version} vs v{treatment_version}",
        detail={"planned_n": computed_n, "alpha": alpha, "mde": mde},
    ))
    return experiment


def start(store: Store, experiment_id: str, *, actor: str = "unknown") -> Experiment:
    experiment = _require(store, experiment_id)
    if experiment.status != "draft":
        raise ExperimentError(f"experiment is {experiment.status}, not draft")

    running = store.running_for_prompt(experiment.prompt_id)
    if running:
        # Two live experiments on one prompt means every observation is confounded by
        # the other's assignment, and neither result is interpretable.
        raise ExperimentError(
            f"{running.id!r} is already running on prompt {experiment.prompt_id!r} — "
            "stop it first; concurrent experiments on one prompt confound each other"
        )

    experiment.status = "running"
    experiment.started_at = datetime.now(timezone.utc)
    store.save_experiment(experiment)
    store.audit(AuditEntry(actor=actor, action="start_experiment", subject=experiment_id,
                           reason="plan frozen at start"))
    return experiment


def amend(store: Store, experiment_id: str, **changes) -> Experiment:
    """Edit a DRAFT experiment. Refuses once it is running."""
    experiment = _require(store, experiment_id)
    if experiment.status != "draft":
        frozen = [f for f in changes if f in FROZEN_FIELDS]
        if frozen:
            raise ExperimentError(
                f"cannot change {frozen} on a {experiment.status} experiment — the sequential "
                "boundary is calibrated against the registered plan, so amending it mid-flight "
                "invalidates the test rather than adjusting it"
            )
    for key, value in changes.items():
        setattr(experiment, key, value)
    store.save_experiment(experiment)
    return experiment


def stop(
    store: Store, experiment_id: str, reason: StopReason, *, actor: str = "unknown", note: str = ""
) -> Experiment:
    experiment = _require(store, experiment_id)
    experiment.status = "completed" if reason in ("winner", "no_difference") else "stopped"
    experiment.stop_reason = reason
    experiment.stopped_at = datetime.now(timezone.utc)
    store.save_experiment(experiment)
    store.audit(AuditEntry(actor=actor, action="stop_experiment", subject=experiment_id,
                           reason=note or reason))
    return experiment


def check_and_autostop(store: Store, experiment_id: str, *, actor: str = "auto") -> tuple[Experiment, str]:
    """Evaluate the guardrail and the sequential boundary, stopping if either fires.

    Only these two stop an experiment automatically. A variant merely trailing on the
    metric never does — that is what the boundary is for, and stopping a losing arm
    early on a look that has not cleared it is exactly the peeking error.
    """
    experiment = _require(store, experiment_id)
    if not experiment.is_live:
        return experiment, f"experiment is {experiment.status}"

    result = analyse(store, experiment)

    if result.guardrail_breached:
        stop(store, experiment_id, "error_guardrail", actor=actor, note=result.guardrail_detail)
        return _require(store, experiment_id), f"STOPPED — {result.guardrail_detail}"

    if result.verdict.crossed:
        winner = (
            experiment.treatment_version
            if result.treatment.mean > result.control.mean
            else experiment.control_version
        )
        if not experiment.metric.higher_is_better:
            winner = (
                experiment.control_version
                if result.treatment.mean > result.control.mean
                else experiment.treatment_version
            )
        experiment = stop(store, experiment_id, "winner", actor=actor, note=result.verdict.reason)
        experiment.winner_version = winner
        store.save_experiment(experiment)
        return experiment, f"WINNER — v{winner}. {result.verdict.reason}"

    if result.verdict.look and result.verdict.look >= len(experiment.look_fractions):
        stop(store, experiment_id, "no_difference", actor=actor, note=result.verdict.reason)
        return _require(store, experiment_id), f"NO WINNER — {result.verdict.reason}"

    return experiment, result.verdict.reason


def promote_winner(store: Store, experiment_id: str, *, actor: str, reason: str = "") -> int:
    experiment = _require(store, experiment_id)
    if experiment.winner_version is None:
        raise ExperimentError("no winner declared — promote only after the boundary is crossed")
    set_active(
        store, experiment.prompt_id, experiment.winner_version,
        actor=actor, reason=reason or f"winner of experiment {experiment_id}",
    )
    return experiment.winner_version


def _require(store: Store, experiment_id: str) -> Experiment:
    experiment = store.get_experiment(experiment_id)
    if not experiment:
        raise ExperimentError(f"no experiment {experiment_id!r}")
    return experiment
