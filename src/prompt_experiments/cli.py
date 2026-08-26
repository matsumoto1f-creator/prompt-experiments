"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_experiments import experiments as exp_ops
from prompt_experiments import registry, simulate
from prompt_experiments.analysis import analyse
from prompt_experiments.models import MetricSpec
from prompt_experiments.providers import MockProvider
from prompt_experiments.stats import naive_peeking_error_rate, obf_critical_value
from prompt_experiments.stats import required_n_proportions, mde_at_n
from prompt_experiments.store import DEFAULT_DB, Store


# ---- prompts -------------------------------------------------------------
def cmd_prompt_new(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        registry.create_prompt(store, a.id, a.name, a.description)
    print(f"created prompt {a.id}")
    return 0


def cmd_prompt_version(a: argparse.Namespace) -> int:
    system = Path(a.file).read_text() if a.file else a.system
    if not system:
        print("[error] provide --system or --file", file=sys.stderr)
        return 2
    with Store(a.db) as store:
        version = registry.add_version(
            store, a.id, system, message=a.message, author=a.author,
            model=a.model, activate=a.activate, reason=a.reason,
        )
    print(f"created {version.ref}  sha={version.content_sha}"
          + ("  (now active)" if a.activate else ""))
    return 0


def cmd_prompt_versions(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        prompt = store.get_prompt(a.id)
        if not prompt:
            print(f"[error] no prompt {a.id!r}", file=sys.stderr)
            return 2
        versions = store.versions(a.id)
    print(f"{prompt.name}  ({len(versions)} versions, active: "
          f"{'v' + str(prompt.active_version) if prompt.active_version else 'none'})\n")
    print(f"{'':2}{'ver':<6}{'sha':<14}{'author':<12}{'created':<21}message")
    for v in versions:
        mark = "* " if v.version == prompt.active_version else "  "
        print(f"{mark}v{v.version:<5}{v.content_sha:<14}{v.author:<12}"
              f"{v.created_at.isoformat()[:19]:<21}{v.message}")
    print("\n* = serving production")
    return 0


def cmd_prompt_diff(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        text = registry.diff(store, a.id, a.left, a.right)
    print(text or "(identical)")
    return 0


def cmd_prompt_activate(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        registry.set_active(store, a.id, a.version, actor=a.actor, reason=a.reason)
    print(f"{a.id}@v{a.version} is now serving production")
    return 0


def cmd_log(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        entries = store.audit_log(limit=a.limit, subject=a.subject)
    if not entries:
        print("no audit entries")
        return 0
    for e in entries:
        print(f"{e.at.isoformat()[:19]}  {e.actor:<10}{e.action:<18}{e.subject:<22}{e.reason}")
    return 0


# ---- experiments ---------------------------------------------------------
def cmd_exp_create(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        experiment = exp_ops.create_experiment(
            store, a.id, a.prompt, a.control, a.treatment,
            metric=MetricSpec(name=a.metric, kind=a.kind),
            baseline=a.baseline, mde=a.mde, alpha=a.alpha,
            traffic_split=a.split, planned_n=a.planned_n,
        )
    print(f"created {experiment.id}: v{a.control} vs v{a.treatment}")
    print(f"  planned n     {experiment.planned_n} per arm"
          + ("" if a.planned_n else f"  (to detect {a.mde:.0%} from a {a.baseline:.0%} baseline at 80% power)"))
    print(f"  looks         {', '.join(f'{t:.0%}' for t in experiment.look_fractions)} of plan")
    print(f"  alpha         {experiment.alpha}")
    print("\nThe plan is frozen once started. That is what the boundary is calibrated against.")
    return 0


def cmd_exp_start(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        exp_ops.start(store, a.id, actor=a.actor)
    print(f"{a.id} is running — traffic is now split")
    return 0


def cmd_exp_status(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        experiment = store.get_experiment(a.id)
        if not experiment:
            print(f"[error] no experiment {a.id!r}", file=sys.stderr)
            return 2
        result = analyse(store, experiment)
    _print_status(result)
    return 0


def cmd_exp_run(a: argparse.Namespace) -> int:
    """Drive synthetic traffic through the real serving path against a known truth."""
    rates = {a.control_version: a.control_rate, a.treatment_version: a.treatment_rate}
    provider = MockProvider(true_rates=rates)
    variables = dict(pair.split("=", 1) for pair in a.var) if a.var else {}
    with Store(a.db) as store:
        trace = simulate.run(store, a.id, provider, n_units=a.n,
                             check_every=a.check_every, variables=variables)
        experiment = store.get_experiment(a.id)
        result = analyse(store, experiment)
    for n, message in trace.checkpoints:
        print(f"  {n:>5} served  {message}")
    print()
    _print_status(result)
    return 0


def cmd_exp_stop(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        exp_ops.stop(store, a.id, "cancelled", actor=a.actor, note=a.reason)
    print(f"{a.id} stopped: {a.reason}")
    return 0


def cmd_exp_promote(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        version = exp_ops.promote_winner(store, a.id, actor=a.actor, reason=a.reason)
    print(f"promoted v{version} to active")
    return 0


def cmd_exp_list(a: argparse.Namespace) -> int:
    with Store(a.db) as store:
        rows = store.list_experiments()
    if not rows:
        print("no experiments")
        return 0
    print(f"{'id':<14}{'prompt':<14}{'arms':<12}{'status':<12}{'winner':<8}reason")
    for e in rows:
        print(f"{e.id:<14}{e.prompt_id:<14}{'v'+str(e.control_version)+' vs v'+str(e.treatment_version):<12}"
              f"{e.status:<12}{('v'+str(e.winner_version)) if e.winner_version else '—':<8}{e.stop_reason}")
    return 0


def _print_status(result) -> None:  # noqa: ANN001
    e = result.experiment
    print(f"experiment {e.id}   {e.status}   metric: {e.metric.name} ({e.metric.kind})")
    print(f"  control   v{result.control.version}  n={result.control.n:<6} {result.control_interval}"
          f"   errors {result.control.error_rate:.1%}   p95 {result.control.latency_p95:.0f}ms")
    print(f"  treatment v{result.treatment.version}  n={result.treatment.n:<6} {result.treatment_interval}"
          f"   errors {result.treatment.error_rate:.1%}   p95 {result.treatment.latency_p95:.0f}ms")
    print(f"  effect    {result.effect}   ({result.test_name}, raw p={result.p_value:.4f})")
    print(f"  guardrail {result.guardrail_detail}")
    print(f"  cost      ${result.control.cost_usd + result.treatment.cost_usd:.4f}")
    print(f"\n  {result.recommendation}")
    if not result.verdict.crossed and result.verdict.look:
        print(f"\n  Raw p is shown for information only. The decision is the boundary: "
              f"|z| {abs(result.verdict.z):.2f} vs {result.verdict.boundary_z:.2f} at this look.")


# ---- statistics ----------------------------------------------------------
def cmd_stats_peeking(a: argparse.Namespace) -> int:
    print("Two IDENTICAL variants. No real difference. How often does each approach")
    print(f"declare a winner at alpha={a.alpha}?\n")
    print(f"{'looks':>6}{'naive (test at alpha every look)':>36}{'this platform':>18}")
    for k in (1, 2, 4, 10, 20):
        fractions = tuple((i + 1) / k for i in range(k))
        naive = naive_peeking_error_rate(a.alpha, fractions, draws=a.draws)
        print(f"{k:>6}{naive:>35.1%}{a.alpha:>18.1%}")
    print("\nThe right-hand column is fixed by construction: the boundary is calibrated")
    print("so that crossing it at ANY planned look has probability alpha in total.")
    c = obf_critical_value(a.alpha, (0.25, 0.5, 0.75, 1.0), draws=a.draws)
    print(f"\nFor four looks, c={c:.3f} — boundary z at each look: "
          + ", ".join(f"{c / t**0.5:.2f}" for t in (0.25, 0.5, 0.75, 1.0)))
    return 0


def cmd_stats_power(a: argparse.Namespace) -> int:
    print(f"baseline {a.baseline:.0%}, alpha {a.alpha}, power {a.power:.0%}\n")
    print(f"{'detect':>10}{'per arm':>12}{'total':>12}")
    for mde in (0.01, 0.02, 0.03, 0.05, 0.10):
        if a.baseline + mde >= 1:
            continue
        n = required_n_proportions(a.baseline, mde, a.alpha, a.power)
        print(f"{mde:>9.0%}{n:>12,}{n*2:>12,}")
    if a.n:
        print(f"\nWith {a.n:,} per arm, the smallest detectable effect is {mde_at_n(a.baseline, a.n, a.alpha, a.power):.1%}.")
        print("Running below that and concluding 'no difference' is a design that could")
        print("never have seen the difference it was looking for.")
    return 0


# ---- parser --------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prompt-exp", description="Prompt versioning and A/B testing.")
    p.add_argument("--db", default=DEFAULT_DB)
    sub = p.add_subparsers(dest="group", required=True)

    pr = sub.add_parser("prompt", help="the registry").add_subparsers(dest="cmd", required=True)
    n = pr.add_parser("new"); n.add_argument("id"); n.add_argument("name")
    n.add_argument("--description", default=""); n.set_defaults(func=cmd_prompt_new)

    v = pr.add_parser("version", help="add an immutable version")
    v.add_argument("id"); v.add_argument("--system"); v.add_argument("--file")
    v.add_argument("--message", required=True); v.add_argument("--author", default="unknown")
    v.add_argument("--model", default="claude-haiku-4-5")
    v.add_argument("--activate", action="store_true"); v.add_argument("--reason", default="")
    v.set_defaults(func=cmd_prompt_version)

    ls = pr.add_parser("versions"); ls.add_argument("id"); ls.set_defaults(func=cmd_prompt_versions)
    d = pr.add_parser("diff"); d.add_argument("id"); d.add_argument("left", type=int)
    d.add_argument("right", type=int); d.set_defaults(func=cmd_prompt_diff)
    ac = pr.add_parser("activate", help="promote or roll back"); ac.add_argument("id")
    ac.add_argument("version", type=int); ac.add_argument("--actor", default="unknown")
    ac.add_argument("--reason", required=True); ac.set_defaults(func=cmd_prompt_activate)

    ex = sub.add_parser("exp", help="experiments").add_subparsers(dest="cmd", required=True)
    c = ex.add_parser("create")
    c.add_argument("id"); c.add_argument("prompt")
    c.add_argument("control", type=int); c.add_argument("treatment", type=int)
    c.add_argument("--metric", default="quality"); c.add_argument("--kind", default="binary",
                                                                 choices=["binary", "continuous"])
    c.add_argument("--baseline", type=float, default=0.70); c.add_argument("--mde", type=float, default=0.05)
    c.add_argument("--alpha", type=float, default=0.05); c.add_argument("--split", type=float, default=0.5)
    c.add_argument("--planned-n", type=int); c.set_defaults(func=cmd_exp_create)

    s = ex.add_parser("start"); s.add_argument("id"); s.add_argument("--actor", default="unknown")
    s.set_defaults(func=cmd_exp_start)
    st = ex.add_parser("status"); st.add_argument("id"); st.set_defaults(func=cmd_exp_status)

    r = ex.add_parser("run", help="drive synthetic traffic against a known ground truth")
    r.add_argument("id"); r.add_argument("--n", type=int, default=3000)
    r.add_argument("--check-every", type=int, default=250)
    r.add_argument("--control-version", type=int, default=1)
    r.add_argument("--treatment-version", type=int, default=2)
    r.add_argument("--control-rate", type=float, default=0.70)
    r.add_argument("--treatment-rate", type=float, default=0.78)
    r.add_argument("--var", action="append", metavar="KEY=VALUE",
                   help="template variable; repeat per variable the prompt needs")
    r.set_defaults(func=cmd_exp_run)

    sp = ex.add_parser("stop"); sp.add_argument("id"); sp.add_argument("--reason", required=True)
    sp.add_argument("--actor", default="unknown"); sp.set_defaults(func=cmd_exp_stop)
    pm = ex.add_parser("promote"); pm.add_argument("id"); pm.add_argument("--actor", default="unknown")
    pm.add_argument("--reason", default=""); pm.set_defaults(func=cmd_exp_promote)
    el = ex.add_parser("list"); el.set_defaults(func=cmd_exp_list)

    stt = sub.add_parser("stats", help="the statistics, on their own").add_subparsers(dest="cmd", required=True)
    pk = stt.add_parser("peeking", help="what continuous monitoring actually costs")
    pk.add_argument("--alpha", type=float, default=0.05); pk.add_argument("--draws", type=int, default=100_000)
    pk.set_defaults(func=cmd_stats_peeking)
    pw = stt.add_parser("power", help="how many observations the question needs")
    pw.add_argument("--baseline", type=float, default=0.70); pw.add_argument("--alpha", type=float, default=0.05)
    pw.add_argument("--power", type=float, default=0.8); pw.add_argument("--n", type=int)
    pw.set_defaults(func=cmd_stats_power)

    lg = sub.add_parser("log", help="audit trail"); lg.add_argument("--limit", type=int, default=30)
    lg.add_argument("--subject"); lg.set_defaults(func=cmd_log, group="log")

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
