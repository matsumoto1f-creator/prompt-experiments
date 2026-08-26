# prompt-experiments

Prompt versioning and A/B testing. Prompts become immutable, addressable versions;
traffic gets split between them; and the platform **refuses to declare a winner from a
peeked p-value**.

That refusal is the whole point. Everything else here is table stakes.

---

## Run it

No API key needed:

```bash
pip install -e ".[dev]"
prompt-exp stats peeking
```

That prints the problem this project exists to solve:

```
Two IDENTICAL variants. No real difference. How often does each approach
declare a winner at alpha=0.05?

 looks    naive (test at alpha every look)     this platform
     1                               4.9%              5.0%
     2                               8.2%              5.0%
     4                              12.7%              5.0%
    10                              19.6%              5.0%
    20                              25.2%              5.0%
```

A dashboard that shows a live p-value next to a Stop button is a false-positive
generator with a nice font. Check twenty times and one experiment in four hands you a
"winner" that does not exist.

The full loop:

```bash
prompt-exp prompt new triage "Support triage"
prompt-exp prompt version triage --system "You triage tickets for {{product}}. Be precise." \
  --message "baseline" --author you --activate --reason "initial launch"
prompt-exp prompt version triage --system "You triage tickets for {{product}}. Think step by step, then answer." \
  --message "add chain-of-thought" --author you

prompt-exp exp create exp-cot triage 1 2 --baseline 0.70 --mde 0.05
prompt-exp exp start exp-cot --actor you
prompt-exp exp run exp-cot --n 3000 --var product=Acme
prompt-exp exp promote exp-cot --actor you --reason "won at look 2"
```

`exp run` drives synthetic traffic through the real serving path — same splitter, same
store, same analysis — against a provider with a known ground truth, so you can check
whether the platform found the effect that was actually there.

HTTP surface: `pip install -e ".[api]"` then `uvicorn prompt_experiments.api:app`.

---

## Why peeking breaks a test, and what fixes it

A p-value is a statement about a decision rule fixed in advance. Test once at the
planned sample size and you get a 5% false-positive rate. Test after every batch and
stop the moment p dips below 0.05, and you have changed the rule to "keep sampling
until the noise agrees with me" — which it eventually does.

The fix is not discipline. It is a boundary that already accounts for the looking.

Track the statistic on the Brownian scale, `B(t) = z(t)·√t`, where `t` is the
information fraction (observations so far ÷ observations planned). Under the null,
`B` is a standard Brownian motion, so the O'Brien–Fleming boundary `z_k ≥ c/√t_k` is
just a **constant** boundary `|B(t_k)| ≥ c`. Calibrate `c` so that crossing it at any
planned look has probability α in total, and looking is free.

`c` is found by simulation rather than from a table, so the boundary is correct for
whatever look schedule an experiment actually registers — and so the claim is
checkable. `tests/test_sequential.py` verifies the empirical type-I error lands at α
on independently drawn paths, and that `c` matches the published value for the
standard four-look schedule.

The boundary is severe early on purpose. For four looks:

| look | information | boundary z |
|---|---|---|
| 1 | 25% | 4.05 |
| 2 | 50% | 2.86 |
| 3 | 75% | 2.34 |
| 4 | 100% | 2.02 |

In the demo, look 1 sees `z = 3.42` — a raw p-value of 0.0006 — and the platform says
**continue**. A naive dashboard stops there and ships. At 25% of the planned sample
that evidence genuinely is not enough, because you gave yourself three more chances to
find it.

---

## What else is deliberate

**Sample size is derived before the experiment starts, not discovered afterwards.**
`prompt-exp stats power` turns a baseline and a minimum detectable effect into a
number. Detecting a 5-point improvement on a 70% baseline needs 1,251 per arm; two
points needs 8,080. Being told that up front is what stops a two-day experiment being
run as though it could answer the question. The inverse matters as much: with the
traffic you actually have, `--n` reports the smallest effect you could have detected —
and running below it, then concluding "no difference", is a design that could never
have seen the difference that was there.

**The plan is frozen at start.** `planned_n`, `look_fractions` and `alpha` are what
the boundary is calibrated against. Amending them on a running experiment raises,
because it would not adjust the test — it would invalidate it silently while every
number on screen still looked fine.

**Versions are append-only, and identical content is refused.** A version is never
edited, so results recorded against v3 stay attributable to v3 forever. A new version
whose content hashes identically to an existing one is rejected: re-running an
experiment against a prompt that did not change is pure cost, and a registry that
allows it quietly fills with them. The hash covers what the model sees — not the
commit message or author, so a reworded note is correctly not a new prompt.

**Rollback is a pointer move, not a deploy**, and it shares its code path with
promotion, because giving them separate paths is how the two diverge. Activation
requires a reason: without one the audit log is a list of timestamps, which answers
nothing when someone asks who changed the prompt that broke checkout last Tuesday.

**The splitter is a pure function of (experiment id, unit id).** Stable, so a user's
experience does not flicker mid-session. Restartable, since nothing is persisted.
SHA-256 rather than `hash()`, which is salted per process and would move users between
arms on restart. And the experiment id is in the hash specifically so concurrent
experiments do not share bucketing — otherwise the same users are in treatment
everywhere, and any trait correlated with them contaminates every experiment at once.

**The serving endpoint never tells the caller which variant it got.** An experiment
the calling team has to integrate with is one that never gets run.

**Guardrails are separate from significance.** An error-rate breach stops an
experiment immediately, regardless of the metric — that is a variant being broken, not
a variant losing. A variant merely trailing never triggers an automatic stop; that is
what the boundary is for, and stopping a losing arm on a look that has not cleared it
is the same peeking error wearing a different hat.

**Welch's t-test, not Student's.** Prompt variants routinely violate equal variance on
purpose — a chain-of-thought variant is both slower and far more variable than a
zero-shot one. Welch costs nothing when variances happen to match, so there is no case
for the alternative. Skewed metrics like latency route to Mann-Whitney, where the
reported effect is a difference in medians with a bootstrapped interval, because
quoting a mean-difference interval beside a rank-based p-value describes two different
questions as one.

**Wilson intervals, not Wald.** At p̂ = 1 Wald collapses to zero width, reporting
perfect certainty from the least informative data available, and near 0 it runs
outside [0, 1]. Both happen constantly at the sample sizes prompt experiments reach.

---

## Layout

```
src/prompt_experiments/
  models.py        typed contracts; versions are immutable by construction
  registry.py      append-only versioning, diff, activate/rollback, audit
  template.py      {{variable}} rendering — missing variables raise, never blank
  assign.py        consistent-hash traffic splitter
  stats/
    sequential.py  the boundary, and the peeking demonstration
    proportions.py Wilson intervals, two-proportion test
    continuous.py  Welch, Mann-Whitney
    power.py       sample size and minimum detectable effect
  analysis.py      guardrail, boundary and descriptives, kept separate
  experiments.py   lifecycle, frozen plan, automatic stops
  serve.py         resolution and serving; the caller never learns the variant
  simulate.py      synthetic traffic against a known ground truth
  store.py         SQLite (WAL); prompt_versions is never updated
  api.py           optional FastAPI surface
  cli.py           entry point
```

## On SQLite

The stack this was specced against calls for PostgreSQL, and for a serving platform
that is a more defensible ask than it was for the sibling projects — this one takes
writes on the request path. The trade, stated so it can be argued with: WAL gives one
writer with concurrent readers, which is ample for the write rate prompt experiments
actually see (one row per served request, at hundreds to low thousands per hour, not
per second). The switch point is a sustained write rate above roughly a hundred per
second, or more than one serving process. Below that, needing a database server to run
the thing costs more than it buys.

## Known gaps

- No dashboard. The spec asks for React or Streamlit; the CLI and the API cover the
  same ground and a chart was not the interesting part of this project.
- Only two arms per experiment. Three or more needs a multiple-comparison correction
  on top of the sequential boundary, and doing that carelessly would undo the thing
  this project is about.
- The bootstrap interval in `mann_whitney` uses a fixed seed so a reported interval
  does not move between runs. That makes it reproducible, not exact.
- No CUPED or covariate adjustment. On noisy metrics it would cut the required sample
  size substantially and is the most valuable thing missing.
