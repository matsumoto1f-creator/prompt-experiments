"""Simulation harness for demos and tests.

Runs traffic through the real serving path — the same splitter, the same store, the
same analysis — against a provider with a KNOWN ground truth. That is what makes the
demo mean something: you can check whether the platform found the effect that was
actually there, and whether it invented one that was not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_experiments.experiments import check_and_autostop
from prompt_experiments.providers import MockProvider
from prompt_experiments.serve import serve
from prompt_experiments.store import Store


@dataclass
class Trace:
    checkpoints: list[tuple[int, str]] = field(default_factory=list)
    served: int = 0
    final: str = ""
    stopped_at: int | None = None


def run(
    store: Store,
    experiment_id: str,
    provider: MockProvider,
    n_units: int,
    check_every: int = 100,
    unit_prefix: str = "user",
    variables: dict[str, str] | None = None,
) -> Trace:
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"no experiment {experiment_id!r}")

    trace = Trace()
    for i in range(n_units):
        unit = f"{unit_prefix}-{i}"
        serve(store, provider, experiment.prompt_id, unit,
              user_input="triage this ticket", variables=variables)
        trace.served += 1

        if trace.served % check_every == 0:
            current, message = check_and_autostop(store, experiment_id)
            trace.checkpoints.append((trace.served, message))
            if not current.is_live:
                trace.stopped_at = trace.served
                trace.final = message
                return trace

    current, message = check_and_autostop(store, experiment_id)
    trace.final = message
    if not current.is_live:
        trace.stopped_at = trace.served
    return trace
