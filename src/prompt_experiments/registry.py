"""The prompt registry — git for prompts, with the parts that actually matter.

What "versioned" has to mean to be worth anything:

  append-only     a version is never edited. Results recorded against v3 stay
                  attributable to v3 forever.
  content-aware   a new version whose content hashes identically to an existing one is
                  refused. Re-running an experiment against a prompt that did not
                  change is pure cost, and a registry that lets you do it silently
                  will fill up with them.
  rollback is a pointer move, not a deploy. The whole point of separating the active
                  version from the code is that reverting a bad prompt takes seconds
                  and needs nobody's CI.
  reasoned        activation requires a reason. Without one the audit log is a list of
                  timestamps, which answers nothing.
"""

from __future__ import annotations

import difflib

from prompt_experiments.models import AuditEntry, FewShot, Prompt, PromptVersion
from prompt_experiments.store import Store


class RegistryError(ValueError):
    pass


def create_prompt(store: Store, prompt_id: str, name: str, description: str = "") -> Prompt:
    if store.get_prompt(prompt_id):
        raise RegistryError(f"prompt {prompt_id!r} already exists")
    prompt = Prompt(id=prompt_id, name=name, description=description)
    store.save_prompt(prompt)
    store.audit(AuditEntry(actor="system", action="create_prompt", subject=prompt_id))
    return prompt


def add_version(
    store: Store,
    prompt_id: str,
    system: str,
    *,
    message: str,
    author: str = "unknown",
    few_shot: list[FewShot] | None = None,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
    effort: str = "low",
    activate: bool = False,
    reason: str = "",
) -> PromptVersion:
    prompt = store.get_prompt(prompt_id)
    if not prompt:
        raise RegistryError(f"no prompt {prompt_id!r}")
    if not message.strip():
        raise RegistryError("a version needs a commit message explaining the change")

    candidate = PromptVersion(
        prompt_id=prompt_id,
        version=store.next_version_number(prompt_id),
        system=system,
        few_shot=few_shot or [],
        model=model,
        max_tokens=max_tokens,
        effort=effort,  # type: ignore[arg-type]
        message=message,
        author=author,
    )

    for existing in store.versions(prompt_id):
        if existing.content_sha == candidate.content_sha:
            raise RegistryError(
                f"identical content to v{existing.version} (sha {existing.content_sha}). "
                "Nothing to test — change the prompt or reuse the existing version."
            )

    store.add_version(candidate)
    store.audit(AuditEntry(
        actor=author, action="add_version", subject=candidate.ref,
        reason=message, detail={"sha": candidate.content_sha},
    ))

    if activate:
        set_active(store, prompt_id, candidate.version, actor=author,
                   reason=reason or f"activated on creation: {message}")
    return candidate


def set_active(store: Store, prompt_id: str, version: int, *, actor: str, reason: str) -> None:
    """Point production at a version. This is both promotion and rollback — there is
    no separate rollback path, because a rollback is just an activation of something
    older and giving it its own code path invites the two to diverge."""
    prompt = store.get_prompt(prompt_id)
    if not prompt:
        raise RegistryError(f"no prompt {prompt_id!r}")
    if not store.get_version(prompt_id, version):
        raise RegistryError(f"no version v{version} of {prompt_id!r}")
    if not reason.strip():
        raise RegistryError("activation requires a reason — an unexplained change is not an audit trail")

    previous = prompt.active_version
    store.set_active(prompt_id, version)
    store.audit(AuditEntry(
        actor=actor,
        action="rollback" if previous is not None and version < previous else "activate",
        subject=f"{prompt_id}@v{version}",
        reason=reason,
        detail={"from": previous, "to": version},
    ))


def active_version(store: Store, prompt_id: str) -> PromptVersion | None:
    prompt = store.get_prompt(prompt_id)
    if not prompt or prompt.active_version is None:
        return None
    return store.get_version(prompt_id, prompt.active_version)


def diff(store: Store, prompt_id: str, left: int, right: int) -> str:
    """Unified diff of what the model sees between two versions."""
    a = store.get_version(prompt_id, left)
    b = store.get_version(prompt_id, right)
    if not a or not b:
        raise RegistryError(f"both v{left} and v{right} of {prompt_id!r} must exist")

    def lines(v: PromptVersion) -> list[str]:
        out = [f"model: {v.model}", f"max_tokens: {v.max_tokens}", f"effort: {v.effort}", "system:"]
        out += v.system.splitlines()
        for index, shot in enumerate(v.few_shot):
            out += [f"few_shot[{index}].input: {shot.input}", f"few_shot[{index}].output: {shot.output}"]
        return out

    return "\n".join(difflib.unified_diff(
        lines(a), lines(b), fromfile=a.ref, tofile=b.ref, lineterm="", n=2
    ))
