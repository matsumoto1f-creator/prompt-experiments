"""HTTP surface. Optional extra: `pip install -e ".[api]"`, then `uvicorn
prompt_experiments.api:app`.

The important endpoint is `POST /v1/completions`, and the important thing about it is
what it does NOT return: which variant served the request. The caller passes a prompt
id, their variables and a stable unit id, and gets a completion. An experiment the
calling team has to integrate with is an experiment that never gets run.

The variant IS reported on `/v1/experiments/{id}` — to the experiment owner, not to
the request path.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prompt_experiments import experiments as exp_ops
from prompt_experiments import registry
from prompt_experiments.analysis import analyse
from prompt_experiments.providers import get_provider
from prompt_experiments.serve import ServingError, serve
from prompt_experiments.store import DEFAULT_DB, Store
from prompt_experiments.template import MissingVariables

app = FastAPI(title="prompt-experiments", version="0.1.0")

DB = os.environ.get("PROMPT_EXPERIMENTS_DB", DEFAULT_DB)
PROVIDER = os.environ.get("PROMPT_EXPERIMENTS_PROVIDER", "mock")


@contextmanager
def _store():
    store = Store(DB)
    try:
        yield store
    finally:
        store._conn.close()  # noqa: SLF001


class CompletionRequest(BaseModel):
    prompt_id: str
    unit_id: str = Field(description="stable user or session id — the split is hashed on it")
    input: str
    variables: dict[str, str] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    text: str
    errored: bool
    latency_ms: float
    # Deliberately no `version` and no `experiment_id`. See the module docstring.


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(request: CompletionRequest) -> CompletionResponse:
    with _store() as store:
        try:
            result = serve(
                store, get_provider(PROVIDER), request.prompt_id,
                request.unit_id, request.input, request.variables,
            )
        except MissingVariables as exc:
            raise HTTPException(422, str(exc)) from exc
        except ServingError as exc:
            raise HTTPException(409, str(exc)) from exc
    return CompletionResponse(text=result.text, errored=result.errored, latency_ms=result.latency_ms)


@app.get("/v1/prompts/{prompt_id}/versions")
def list_versions(prompt_id: str) -> dict:
    with _store() as store:
        prompt = store.get_prompt(prompt_id)
        if not prompt:
            raise HTTPException(404, f"no prompt {prompt_id!r}")
        return {
            "prompt": prompt.model_dump(mode="json"),
            "versions": [
                {**v.model_dump(mode="json"), "content_sha": v.content_sha}
                for v in store.versions(prompt_id)
            ],
        }


@app.get("/v1/prompts/{prompt_id}/diff")
def diff(prompt_id: str, left: int, right: int) -> dict:
    with _store() as store:
        try:
            return {"diff": registry.diff(store, prompt_id, left, right)}
        except registry.RegistryError as exc:
            raise HTTPException(404, str(exc)) from exc


class ActivateRequest(BaseModel):
    version: int
    actor: str
    reason: str


@app.post("/v1/prompts/{prompt_id}/active")
def activate(prompt_id: str, request: ActivateRequest) -> dict:
    """Promotion and rollback. A pointer move — no deploy, no CI, seconds not minutes."""
    with _store() as store:
        try:
            registry.set_active(store, prompt_id, request.version,
                                actor=request.actor, reason=request.reason)
        except registry.RegistryError as exc:
            raise HTTPException(400, str(exc)) from exc
    return {"prompt_id": prompt_id, "active_version": request.version}


@app.get("/v1/experiments/{experiment_id}")
def experiment_status(experiment_id: str) -> dict:
    with _store() as store:
        experiment = store.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(404, f"no experiment {experiment_id!r}")
        result = analyse(store, experiment)
        return {
            "experiment": experiment.model_dump(mode="json"),
            "control": result.control.model_dump(),
            "treatment": result.treatment.model_dump(),
            "effect": result.effect,
            "test": result.test_name,
            # Present, and clearly labelled as not being the decision.
            "raw_p_value_informational_only": result.p_value,
            "verdict": {
                "look": result.verdict.look,
                "information_fraction": result.verdict.information_fraction,
                "z": result.verdict.z,
                "boundary_z": result.verdict.boundary_z,
                "crossed": result.verdict.crossed,
                "reason": result.verdict.reason,
            },
            "guardrail_breached": result.guardrail_breached,
            "recommendation": result.recommendation,
        }


@app.post("/v1/experiments/{experiment_id}/check")
def check(experiment_id: str) -> dict:
    """Evaluate guardrail and boundary, stopping if either fires."""
    with _store() as store:
        try:
            experiment, message = exp_ops.check_and_autostop(store, experiment_id)
        except exp_ops.ExperimentError as exc:
            raise HTTPException(404, str(exc)) from exc
    return {"status": experiment.status, "stop_reason": experiment.stop_reason,
            "winner_version": experiment.winner_version, "message": message}


@app.get("/v1/audit")
def audit(limit: int = 50, subject: str | None = None) -> dict:
    with _store() as store:
        return {"entries": [e.model_dump(mode="json") for e in store.audit_log(limit, subject)]}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "provider": PROVIDER, "db": DB}
