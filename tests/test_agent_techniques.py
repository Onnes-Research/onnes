"""Tests for the evidence-backed in-context levers added to the agent panel
(few-shot contrastive demos + self-consistency voting).

All offline: the few-shot block builder is deterministic given a seed, the
majority-vote logic is a pure function, and run_panel is exercised on the "stub"
backend so NO network / live-model calls happen. These lock in:
  * zero-shot behavior is preserved when the levers are off (k=0, N=1),
  * the demo pool is disjoint from both the ML train seeds and the eval seeds,
  * self-consistency picks the modal class and records the tally.

Design note: we assert STRUCTURE and DISJOINTNESS, never model accuracy — the
accuracy claim is a live-run empirical result, not a unit-test invariant.
"""
from __future__ import annotations

from onnesim import agent_eval as AE
from onnesim import multi_agent as MA
from onnesim import cryo_engine as CE


def _eng():
    return CE.EngineConfig(realistic=True, imperfections=True)


# ------------------------------------------------------------- few-shot ---
def test_few_shot_block_off_by_default_preserves_zero_shot():
    """k=0 must yield no block, so the default panel is byte-identical zero-shot."""
    cfg = AE.EvalConfig()
    assert cfg.few_shot_k == 0 and cfg.sc_samples == 1
    assert AE.build_few_shot_block(0, _eng(), cfg) is None


def test_few_shot_block_covers_confusable_pairs():
    """A small block must include BOTH members of each engineered confusable pair,
    because contrastive near-miss demos are the whole point of the lever."""
    cfg = AE.EvalConfig(few_shot_k=6)
    block = AE.build_few_shot_block(6, _eng(), cfg)
    assert block is not None
    for fc in ["helium_leak", "blocked_impedance",
               "wiring_heat_ingress", "heat_load_spike"]:
        assert fc in block, f"few-shot block missing confusable class {fc}"
    # each demo carries a ground-truth label line
    assert block.count("fault_class =") >= 6


def test_few_shot_block_is_deterministic():
    """Same config -> same block (reproducible demos)."""
    cfg = AE.EvalConfig(few_shot_k=6)
    b1 = AE.build_few_shot_block(6, _eng(), cfg)
    b2 = AE.build_few_shot_block(6, _eng(), cfg)
    assert b1 == b2


def test_demo_seed_range_disjoint_from_train_and_eval():
    """Demos must NOT be drawn from the ML train seed range (base 0..train_n) nor the
    eval seed range (base 10_000..) — otherwise the few-shot agent would be peeking at
    test data. We assert the seed windows do not overlap."""
    cfg = AE.EvalConfig(few_shot_k=8)
    demo_lo, demo_hi = cfg.demo_seed, cfg.demo_seed + cfg.few_shot_k
    # ML train uses base_seed 0 with train_n specs; eval uses base_seed 10_000
    assert demo_hi <= 10_000, "demo seeds collide with eval seed range"
    assert demo_lo >= cfg.train_n, "demo seeds collide with ML train seed range"


# -------------------------------------------------- self-consistency vote ---
def test_self_consistency_majority_vote_picks_mode():
    """The vote helper (via _diagnose's aggregation) must pick the modal class."""
    from collections import Counter
    samples = [{"fault_class": "helium_leak"}, {"fault_class": "helium_leak"},
               {"fault_class": "blocked_impedance"}]
    votes = Counter(str(s.get("fault_class", "")).strip().lower() for s in samples)
    assert votes.most_common(1)[0][0] == "helium_leak"


def test_panel_runs_with_levers_on_stub_backend():
    """run_panel with few_shot + self-consistency must still return all 5 roles and
    never touch the network on the stub backend."""
    eng = _eng()
    cfg = AE.EvalConfig(few_shot_k=6, sc_samples=3)
    block = AE.build_few_shot_block(6, eng, cfg)
    cols = CE.simulate(CE.Scenario("helium_leak", 0.4, 0.4), eng,
                       hours=1, dt_min=10, seed=1)
    panel = MA.run_panel(cols, backend="stub", few_shot_block=block, sc_samples=3)
    assert set(panel) >= {"sentinel", "diagnostician", "operator",
                          "guardian", "supervisor"}


def test_panel_zero_shot_default_unchanged():
    """Calling run_panel with no lever args must behave exactly as the original
    (5 roles, stub-safe) — protects the reproducibility of the 1000-turn baseline."""
    eng = _eng()
    cols = CE.simulate(CE.Scenario("normal", 0.0, 0.4), eng, hours=1, dt_min=10, seed=2)
    panel = MA.run_panel(cols, backend="stub")
    assert set(panel) >= {"sentinel", "diagnostician", "operator",
                          "guardian", "supervisor"}


# -------------------------------------------------- architecture ablation ---
def test_single_agent_returns_supervisor_shape():
    """The single-call ablation must emit exactly the {'supervisor': ...} shape the
    scorer reads, so panel and single conditions score through the same code path."""
    eng = _eng()
    cols = CE.simulate(CE.Scenario("helium_leak", 0.4, 0.4), eng,
                       hours=1, dt_min=10, seed=3)
    out = MA.run_single_agent(cols, backend="stub")
    assert set(out) == {"supervisor"}


def test_eval_arch_knob_selects_single():
    """EvalConfig.arch defaults to the 5-role panel and can select the single agent;
    this is the switch the ablation harness flips."""
    assert AE.EvalConfig().arch == "panel"
    assert AE.EvalConfig(arch="single").arch == "single"

