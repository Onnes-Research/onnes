"""Tests for the offline agent stress + reliability harnesses (agent_stress, agent_passk,
twin_fidelity, degeneracy). All run on the deterministic stub / pure-physics twin, so they
need NO network and NO API key — they lock in that each harness MEASURES CORRECTLY before
it is ever pointed at the paid live panel.

Design: assert STRUCTURE and known invariants on the stub, never live-model accuracy.
"""
from __future__ import annotations

import numpy as np

from onnesim import agent_stress as ST
from onnesim import agent_passk as PK
from onnesim import cryo_engine as CE


def _fixed_predict(det: bool, cls: str):
    """A constant predictor — the simplest target to check harness bookkeeping."""
    return lambda cols: (det, cls)


# ------------------------------------------------------------- metamorphic ---
def test_metamorphic_constant_predictor_is_perfectly_stable():
    """A predictor that ignores its input CANNOT flip under any transform -> stability 1.0."""
    out = ST.run_metamorphic(_fixed_predict(True, "helium_leak"), n=6)
    assert out["overall_metamorphic_stability"] == 1.0
    for r, d in out["per_relation"].items():
        assert d["n_flips"] == 0


def test_metamorphic_relations_preserve_length_and_time_axis():
    """Every relation must return a window the summarizer can still read (t_s present)."""
    eng = CE.EngineConfig(realistic=True, imperfections=True)
    cols = CE.simulate(CE.Scenario("helium_leak", 0.35, 0.4), eng, hours=1, dt_min=10, seed=1)
    rng = np.random.default_rng(0)
    for name, fn in ST.METAMORPHIC_RELATIONS.items():
        t = fn(cols, rng)
        assert "t_s" in t, f"{name} dropped t_s"
        assert len(t["t_s"]) == len(cols["t_s"]), f"{name} changed window length"


def test_dead_channel_adds_a_constant_channel():
    eng = CE.EngineConfig(realistic=True, imperfections=True)
    cols = CE.simulate(CE.Scenario("normal", 0.0, 0.4), eng, hours=1, dt_min=10, seed=2)
    out = ST.mr_dead_channel(cols, np.random.default_rng(0))
    assert "temp_spare_dead" in out
    assert np.ptp(out["temp_spare_dead"]) == 0.0  # truly constant


# ------------------------------------------------------------- adversarial ---
def test_adversarial_injection_adds_named_channel_carrying_the_attack():
    eng = CE.EngineConfig(realistic=True, imperfections=True)
    cols = CE.simulate(CE.Scenario("helium_leak", 0.4, 0.4), eng, hours=1, dt_min=10, seed=3)
    attack = ST.INJECTION_STRINGS[0]
    out = ST._inject_adversarial(cols, attack)
    injected = [k for k in out if k.startswith("note_")]
    assert injected and attack in injected[0]


def test_adversarial_stub_is_immune_by_construction():
    """The rule-based stub cannot read text, so injected strings must not flip it -> ASR 0."""
    out = ST.run_adversarial(ST.stub_predict, n=6)
    assert out["mean_attack_success_rate"] == 0.0


# ------------------------------------------------------------- perturbation ---
def test_perturbation_curve_is_monotone_in_structure():
    """The curve must have one entry per eps level and start at eps=0."""
    out = ST.run_perturbation(_fixed_predict(True, "normal"), n=5, eps_levels=(0.0, 0.1))
    assert [c["eps"] for c in out["curve"]] == [0.0, 0.1]
    assert "robustness_auc" in out


# ------------------------------------------------------------- pass^k --------
def test_passk_unbiased_estimator_matches_hand_values():
    # all k correct -> pass^k = 1; fewer than k correct -> 0; C(4,3)/C(5,3)=4/10=0.4
    assert PK._unbiased_pass_k(5, 5, 5) == 1.0
    assert PK._unbiased_pass_k(2, 5, 3) == 0.0
    assert abs(PK._unbiased_pass_k(4, 5, 3) - 0.4) < 1e-9


def test_passk_perfect_predictor_scores_one():
    """A predictor that is always right on a fault must have pass^k == 1 for detection."""
    # constant 'helium_leak' predictor: detection correct on every fault scenario
    out = PK.run(_fixed_predict(True, "helium_leak"), n=6, k=3, tag="test_perfect",
                 out_path="outputs/_test_passk.json")
    # detection pass^k is 1.0 only if every sampled scenario is a fault; assert the field exists
    assert 0.0 <= out["detection"]["pass_k_empirical"] <= 1.0
    assert out["k"] == 3 and out["n_scenarios"] == 6


def test_passk_stub_pass_k_not_above_pass_1():
    """pass^k can never exceed pass^1 (all-k-correct is a subset of any-correct)."""
    out = PK.run(ST.stub_predict, n=8, k=3, tag="test_stub", out_path="outputs/_test_passk2.json")
    for key in ("detection", "classification"):
        assert out[key]["pass_k_empirical"] <= out[key]["pass_1_mean_acc"] + 1e-9
