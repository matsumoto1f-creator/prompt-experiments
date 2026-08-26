"""Traffic splitting.

Assignment is a pure function of (experiment id, unit id) — no randomness, no state,
no lookup. Three properties fall out of that, all of which matter:

  stable      the same user always sees the same variant, so their experience does
              not flicker between arms mid-session and their observations are not
              a mixture of both.
  independent adding a second experiment does not disturb the first one's split,
              because the experiment id is in the hash.
  restartable a restarted service assigns identically without persisting anything.

The experiment id is salted into the hash for the second property specifically. A
splitter that hashes only the user id gives every concurrent experiment the same
bucketing, so the same users are always in treatment everywhere — and any correlation
between those users and the outcome contaminates every experiment at once.
"""

from __future__ import annotations

import hashlib

BUCKETS = 10_000


def bucket(experiment_id: str, unit_id: str) -> int:
    """Stable bucket in [0, BUCKETS).

    SHA-256 rather than the built-in `hash()`, which is salted per process — the same
    user would land in different arms after a restart.
    """
    digest = hashlib.sha256(f"{experiment_id}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def assign(experiment_id: str, unit_id: str, treatment_share: float) -> bool:
    """True if this unit belongs to the treatment arm."""
    if not 0 < treatment_share < 1:
        raise ValueError("treatment_share must lie strictly between 0 and 1")
    return bucket(experiment_id, unit_id) < treatment_share * BUCKETS
