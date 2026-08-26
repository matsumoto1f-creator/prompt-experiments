"""The serving path.

One call. The caller passes a prompt id, their template variables and a stable unit
id, and gets a completion back. **It never learns which variant it got**, and that is
the design: an experiment the calling code has to know about is one every team has to
integrate with separately, and it will not get adopted.

Resolution order:
  1. A running experiment on this prompt -> the splitter picks the arm.
  2. Otherwise the prompt's active version.
  3. Otherwise an error, because serving an unversioned prompt is how you end up
     unable to explain last Tuesday.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from prompt_experiments.assign import assign
from prompt_experiments.models import Observation, PromptVersion
from prompt_experiments.providers import Provider
from prompt_experiments.registry import active_version
from prompt_experiments.store import Store
from prompt_experiments.template import render


class ServingError(RuntimeError):
    pass


@dataclass
class Served:
    text: str
    version: int
    experiment_id: str | None
    errored: bool
    latency_ms: float
    cost_usd: float


def resolve_version(store: Store, prompt_id: str, unit_id: str) -> tuple[PromptVersion, str | None]:
    experiment = store.running_for_prompt(prompt_id)
    if experiment:
        in_treatment = assign(experiment.id, unit_id, experiment.traffic_split)
        version_number = experiment.treatment_version if in_treatment else experiment.control_version
        version = store.get_version(prompt_id, version_number)
        if not version:
            raise ServingError(f"experiment {experiment.id} references missing v{version_number}")
        return version, experiment.id

    version = active_version(store, prompt_id)
    if not version:
        raise ServingError(f"prompt {prompt_id!r} has no active version and no running experiment")
    return version, None


def serve(
    store: Store,
    provider: Provider,
    prompt_id: str,
    unit_id: str,
    user_input: str,
    variables: dict[str, str] | None = None,
    *,
    score=None,  # noqa: ANN001 - callable(text) -> float, the metric for this request
) -> Served:
    version, experiment_id = resolve_version(store, prompt_id, unit_id)

    # The system prompt is the template; the caller's message is passed through as-is.
    # `strict=True` on purpose: a missing variable raises here rather than rendering a
    # blank where the customer's name should be and getting quietly worse.
    filled = version.model_copy(update={"system": render(version.system, variables or {})})

    completion = provider.complete(filled, user_input, unit_id)

    if experiment_id:
        value = 0.0
        if not completion.errored:
            value = float(score(completion.text)) if score else (1.0 if completion.text == "PASS" else 0.0)
        store.record(Observation(
            id=uuid.uuid4().hex[:12],
            experiment_id=experiment_id,
            unit_id=unit_id,
            version=version.version,
            value=value,
            latency_ms=completion.latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
            errored=completion.errored,
            note=completion.error,
        ))

    return Served(
        text=completion.text,
        version=version.version,
        experiment_id=experiment_id,
        errored=completion.errored,
        latency_ms=completion.latency_ms,
        cost_usd=completion.cost_usd,
    )
