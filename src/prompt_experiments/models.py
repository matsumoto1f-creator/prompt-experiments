"""Typed contracts.

Three objects carry the system:

  PromptVersion   an immutable prompt. Never edited — superseded.
  Experiment      a pre-registered comparison between two versions.
  Observation     one served request and what happened to it.

Immutability of versions is the load-bearing choice. If a version can be edited in
place, then every result recorded against it becomes unattributable, and the audit
trail answers "who changed the prompt that broke the feature last Tuesday" with a
shrug.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ExperimentStatus = Literal["draft", "running", "stopped", "completed", "cancelled"]
StopReason = Literal[
    "winner", "no_difference", "error_guardrail", "quality_guardrail", "cancelled", ""
]
MetricKind = Literal["binary", "continuous"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FewShot(BaseModel):
    input: str
    output: str


class PromptVersion(BaseModel):
    """An immutable prompt version. The unit CI, experiments and rollback all address."""

    prompt_id: str
    version: int
    system: str
    few_shot: list[FewShot] = Field(default_factory=list)
    model: str = "claude-haiku-4-5"
    max_tokens: int = 1024
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    message: str = ""
    author: str = "unknown"
    created_at: datetime = Field(default_factory=_now)

    @property
    def ref(self) -> str:
        return f"{self.prompt_id}@v{self.version}"

    @property
    def content_sha(self) -> str:
        """Hash of what the model actually sees.

        Deliberately excludes the version number, message and author: two versions
        with different commit messages and identical content ARE the same prompt, and
        the registry should be able to say so rather than pretending a no-op edit is
        a change worth re-testing.
        """
        payload = json.dumps(
            {
                "system": self.system,
                "few_shot": [s.model_dump() for s in self.few_shot],
                "model": self.model,
                "max_tokens": self.max_tokens,
                "effort": self.effort,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class Prompt(BaseModel):
    id: str
    name: str
    description: str = ""
    active_version: int | None = None
    created_at: datetime = Field(default_factory=_now)


class AuditEntry(BaseModel):
    """Every state change, with who and why. The reason field is required for the
    actions that can break production — an unexplained rollback is not an audit
    trail, it is a timestamp."""

    at: datetime = Field(default_factory=_now)
    actor: str
    action: str
    subject: str
    reason: str = ""
    detail: dict = Field(default_factory=dict)


class MetricSpec(BaseModel):
    name: str
    kind: MetricKind = "binary"
    higher_is_better: bool = True


class Guardrail(BaseModel):
    """Automatic stop conditions. Distinct from the significance boundary: these are
    about a variant being actively harmful, not about it being worse on the metric."""

    max_error_rate: float = 0.10
    min_observations_before_check: int = 30

    @field_validator("max_error_rate")
    @classmethod
    def _sane_rate(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("max_error_rate must lie in (0, 1]")
        return v


class Experiment(BaseModel):
    """A pre-registered comparison.

    `planned_n` and `look_fractions` are registered before any traffic is served and
    are not editable afterwards — the sequential boundary is calibrated against them,
    so a schedule changed mid-flight silently invalidates the test it was protecting.
    """

    id: str
    prompt_id: str
    control_version: int
    treatment_version: int
    metric: MetricSpec
    traffic_split: float = 0.5          # share routed to treatment
    planned_n: int = 1000               # per arm
    look_fractions: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    alpha: float = 0.05
    baseline_estimate: float = 0.7
    mde: float = 0.05
    guardrail: Guardrail = Field(default_factory=Guardrail)
    status: ExperimentStatus = "draft"
    stop_reason: StopReason = ""
    winner_version: int | None = None
    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    stopped_at: datetime | None = None

    @field_validator("traffic_split")
    @classmethod
    def _split_range(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("traffic_split must lie strictly between 0 and 1")
        return v

    @property
    def is_live(self) -> bool:
        return self.status == "running"


class Observation(BaseModel):
    """One served request and its outcome."""

    id: str
    experiment_id: str
    unit_id: str                    # the user or session the split was hashed on
    version: int
    at: datetime = Field(default_factory=_now)
    value: float = 0.0              # the metric: 1/0 for binary, the number otherwise
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errored: bool = False
    note: str = ""


class ArmSummary(BaseModel):
    version: int
    n: int
    successes: int = 0
    mean: float = 0.0
    errors: int = 0
    cost_usd: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.n if self.n else 0.0
