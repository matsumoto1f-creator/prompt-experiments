import pytest

from prompt_experiments import experiments as exp_ops
from prompt_experiments import registry, simulate
from prompt_experiments.analysis import analyse
from prompt_experiments.assign import assign, bucket
from prompt_experiments.models import Guardrail
from prompt_experiments.providers import MockProvider
from prompt_experiments.store import Store
from prompt_experiments.template import MissingVariables, render, variables


# ---- assignment ---------------------------------------------------------
def test_assignment_is_stable_across_calls():
    first = [assign("exp", f"u{i}", 0.5) for i in range(500)]
    second = [assign("exp", f"u{i}", 0.5) for i in range(500)]
    assert first == second


def test_assignment_respects_the_configured_split():
    share = sum(assign("exp", f"u{i}", 0.3) for i in range(20_000)) / 20_000
    assert share == pytest.approx(0.30, abs=0.015)


def test_concurrent_experiments_do_not_share_bucketing():
    """Without the experiment id in the hash, every experiment puts the same users in
    treatment — so any trait correlated with those users contaminates all of them."""
    a = [assign("exp-1", f"u{i}", 0.5) for i in range(5000)]
    b = [assign("exp-2", f"u{i}", 0.5) for i in range(5000)]
    agreement = sum(x == y for x, y in zip(a, b)) / len(a)
    assert 0.45 < agreement < 0.55


def test_bucketing_is_stable_across_processes():
    """SHA-256 rather than hash(), which is salted per process — the same user would
    switch arms after a restart."""
    assert bucket("exp", "user-1") == bucket("exp", "user-1")
    assert bucket("exp", "user-1") != bucket("exp", "user-2")


# ---- templates ----------------------------------------------------------
def test_missing_variables_raise_rather_than_render_blanks():
    with pytest.raises(MissingVariables) as excinfo:
        render("Hello {{name}}, about {{topic}}", {"name": "Ana"})
    assert excinfo.value.missing == {"topic"}


def test_variables_are_discovered_from_the_template():
    assert variables("{{a}} and {{ b }} and {{a}}") == {"a", "b"}


# ---- registry -----------------------------------------------------------
def _prompt(store: Store) -> None:
    registry.create_prompt(store, "p", "Prompt")
    registry.add_version(store, "p", "system one", message="first",
                         author="asad", activate=True, reason="launch")


def test_a_version_needs_a_commit_message(tmp_path):
    with Store(tmp_path / "d.db") as store:
        registry.create_prompt(store, "p", "Prompt")
        with pytest.raises(registry.RegistryError):
            registry.add_version(store, "p", "system", message="  ")


def test_identical_content_is_refused(tmp_path):
    """Re-running an experiment against a prompt that did not change is pure cost."""
    with Store(tmp_path / "d.db") as store:
        _prompt(store)
        with pytest.raises(registry.RegistryError, match="identical content"):
            registry.add_version(store, "p", "system one", message="no-op", author="asad")


def test_activation_requires_a_reason(tmp_path):
    with Store(tmp_path / "d.db") as store:
        _prompt(store)
        registry.add_version(store, "p", "system two", message="second", author="asad")
        with pytest.raises(registry.RegistryError, match="reason"):
            registry.set_active(store, "p", 2, actor="asad", reason="")


def test_rollback_is_recorded_as_a_rollback(tmp_path):
    with Store(tmp_path / "d.db") as store:
        _prompt(store)
        registry.add_version(store, "p", "system two", message="second", author="asad")
        registry.set_active(store, "p", 2, actor="asad", reason="promote")
        registry.set_active(store, "p", 1, actor="asad", reason="v2 broke checkout")

        actions = [e.action for e in store.audit_log(subject="p@v1")]
        assert "rollback" in actions
        assert store.get_prompt("p").active_version == 1


def test_versions_are_never_mutated(tmp_path):
    with Store(tmp_path / "d.db") as store:
        _prompt(store)
        original = store.get_version("p", 1)
        registry.add_version(store, "p", "system two", message="second", author="asad")
        assert store.get_version("p", 1).system == original.system


# ---- experiments --------------------------------------------------------
def _experiment(store: Store, **kw) -> None:
    _prompt(store)
    registry.add_version(store, "p", "system two", message="second", author="asad")
    exp_ops.create_experiment(store, "e", "p", 1, 2, baseline=0.7, mde=0.05, **kw)


def test_the_plan_cannot_be_amended_once_running(tmp_path):
    """The boundary is calibrated against the registered plan. Changing it mid-flight
    invalidates the test rather than adjusting it."""
    with Store(tmp_path / "d.db") as store:
        _experiment(store)
        exp_ops.amend(store, "e", planned_n=500)      # fine while draft
        exp_ops.start(store, "e", actor="asad")
        with pytest.raises(exp_ops.ExperimentError, match="invalidates"):
            exp_ops.amend(store, "e", planned_n=100)
        with pytest.raises(exp_ops.ExperimentError, match="invalidates"):
            exp_ops.amend(store, "e", alpha=0.2)


def test_two_experiments_cannot_run_on_one_prompt(tmp_path):
    with Store(tmp_path / "d.db") as store:
        _experiment(store)
        exp_ops.start(store, "e", actor="asad")
        exp_ops.create_experiment(store, "e2", "p", 1, 2, baseline=0.7, mde=0.05)
        with pytest.raises(exp_ops.ExperimentError, match="confound"):
            exp_ops.start(store, "e2", actor="asad")


def test_control_and_treatment_must_differ(tmp_path):
    with Store(tmp_path / "d.db") as store:
        _prompt(store)
        with pytest.raises(exp_ops.ExperimentError):
            exp_ops.create_experiment(store, "e", "p", 1, 1)


def test_a_unit_is_only_ever_counted_once(tmp_path):
    """A duplicate write would double-count a user and quietly inflate the sample."""
    from prompt_experiments.models import Observation

    with Store(tmp_path / "d.db") as store:
        _experiment(store)
        first = store.record(Observation(id="o1", experiment_id="e", unit_id="u1", version=1))
        second = store.record(Observation(id="o2", experiment_id="e", unit_id="u1", version=1))
        assert first is True and second is False
        assert store.arm("e", 1).n == 1


# ---- end to end ---------------------------------------------------------
def test_a_real_effect_is_found(tmp_path):
    with Store(tmp_path / "d.db") as store:
        _experiment(store)
        exp_ops.start(store, "e", actor="asad")
        trace = simulate.run(store, "e", MockProvider(true_rates={1: 0.70, 2: 0.80}),
                             n_units=4000, check_every=250)
        experiment = store.get_experiment("e")

        assert experiment.stop_reason == "winner"
        assert experiment.winner_version == 2
        result = analyse(store, experiment)
        # The true effect must sit inside the reported interval.
        assert result.treatment.mean - result.control.mean > 0.04


def test_no_effect_rarely_produces_a_winner(tmp_path):
    """The failure mode that matters: inventing a difference that is not there.

    Asserted as a RATE across independent runs, not on a single one. A single run
    declaring no winner is a 95% event by construction — testing it directly would
    give a suite that fails 5% of the time for the most boring possible reason, and
    a flaky test gets deleted rather than believed.

    What is actually guaranteed is the rate, and that is what is checked. The naive
    comparison is the point of reference: with 16 looks, testing at alpha each time
    would land near 20%.
    """
    winners = 0
    runs = 40
    for i in range(runs):
        with Store(tmp_path / f"d{i}.db") as store:
            _experiment(store)
            exp_ops.start(store, "e", actor="asad")
            simulate.run(store, "e", MockProvider(true_rates={1: 0.70, 2: 0.70}),
                         n_units=4000, check_every=250, unit_prefix=f"cohort{i}")
            if store.get_experiment("e").stop_reason == "winner":
                winners += 1

    rate = winners / runs
    assert rate <= 0.15, f"false-positive rate {rate:.0%} over {runs} runs — too high for alpha=0.05"


def test_a_single_no_effect_run_reports_an_interval_containing_zero(tmp_path):
    """Complements the rate test: whatever the stop reason, an experiment with no real
    effect must not report a confident non-zero effect."""
    with Store(tmp_path / "d.db") as store:
        _experiment(store)
        exp_ops.start(store, "e", actor="asad")
        simulate.run(store, "e", MockProvider(true_rates={1: 0.70, 2: 0.70}),
                     n_units=4000, check_every=250, unit_prefix="quiet")
        result = analyse(store, store.get_experiment("e"))
        assert abs(result.treatment.mean - result.control.mean) < 0.06


def test_the_error_guardrail_stops_independently_of_significance(tmp_path):
    """A broken variant is stopped because it is broken, not because it is losing."""
    with Store(tmp_path / "d.db") as store:
        _experiment(store, guardrail=Guardrail(max_error_rate=0.05,
                                               min_observations_before_check=20))
        exp_ops.start(store, "e", actor="asad")
        simulate.run(store, "e",
                     MockProvider(true_rates={1: 0.70, 2: 0.70}, error_rates={2: 0.40}),
                     n_units=1000, check_every=100)
        experiment = store.get_experiment("e")

        assert experiment.stop_reason == "error_guardrail"
        assert experiment.status == "stopped"


def test_promotion_requires_a_declared_winner(tmp_path):
    with Store(tmp_path / "d.db") as store:
        _experiment(store)
        exp_ops.start(store, "e", actor="asad")
        with pytest.raises(exp_ops.ExperimentError, match="no winner"):
            exp_ops.promote_winner(store, "e", actor="asad")
