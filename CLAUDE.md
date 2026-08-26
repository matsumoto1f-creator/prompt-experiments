# prompt-experiments — working notes

Read README.md first. This file is invariants that are easy to break, and mistakes
already made once.

## Ground rules

- `prompt_versions` is APPEND-ONLY. There is no UPDATE against it anywhere; keep it
  that way or every recorded result becomes unattributable.
- The offline path (`mock` provider, no key, no network) must keep working. Vendor
  imports stay deferred.
- `planned_n`, `look_fractions` and `alpha` are frozen once an experiment is running.
  The boundary is calibrated against them.
- Never log the value of `ANTHROPIC_API_KEY`.

## Lab notes (what not to try)

- **Do not assert a single stochastic run's outcome.** A test that a no-effect
  experiment produces no winner is asserting a 95% event — it fails 5% of the time by
  construction. Measured properly: the full pipeline's false-positive rate is 4.2%
  over 120 independent runs, exactly on target. Assert the RATE across runs, not the
  draw. A flaky test gets deleted rather than believed.
- **Do not conflate case difficulty with statistical certainty** (the same error as in
  eval-dataset-miner, in a different costume). Here: a variant trailing on the metric
  is not a reason to auto-stop. Only a guardrail breach or a crossed boundary stops an
  experiment.
- **The OBF boundary is a CONSTANT on the Brownian scale.** `z_k ≥ c/√t_k` is
  equivalent to `|B(t_k)| ≥ c`. That is what makes calibration by simulation trivial
  and exact — do not reach for a lookup table.
- **Salt the splitter with the experiment id.** Hashing only the unit id gives every
  concurrent experiment identical bucketing. Test:
  `test_concurrent_experiments_do_not_share_bucketing`.
- **`hash()` is salted per process** — SHA-256, or users switch arms on restart.
- **Pool the variance for the test statistic, not for the interval.** The null asserts
  equal proportions so the statistic pools; the interval on the difference must not,
  or it disagrees with the p-value near the boundary.
- **zsh does not word-split unquoted parameters** the way bash does. `D="--db x"; cmd $D`
  passes one argument, not two. Cost a confusing round of CLI "failures".

## Verify a change

```bash
pytest -q
prompt-exp stats peeking --draws 40000
rm -f demo.db && prompt-exp --db demo.db prompt new triage "Support triage" && \
  prompt-exp --db demo.db prompt version triage --system "Triage for {{product}}. Be precise." \
    --message baseline --author you --activate --reason launch
```

On the fixture: `stats peeking` must show naive rising to ~25% at 20 looks while the
right column stays at 5%. The four-look OBF constant is ~2.02. Those moving means the
calibration changed — check it was on purpose.
