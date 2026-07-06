"""
ablation.py — isolate what actually drives the agent's performance.

Three reviewers asked for two ablations the paper was missing:

  A. Technique decomposition — the enhanced panel adds TWO levers at once (contrastive
     few-shot + self-consistency), so the lift can't be attributed. We run each alone:
       zero_shot         : k=0, N=1        (baseline)
       few_shot_only     : k=6, N=1
       self_consistency  : k=0, N=3
       both              : k=6, N=3        (the enhanced panel)

  B. Architecture — the paper cites "The Illusion of Multi-Agent Advantage" yet never
     tests its own 5-role panel against a single agent. We run the SAME conditions with
     arch='single' (one LLM call) vs arch='panel' (five). If the panel doesn't beat one
     call at 5x the cost, the honest move is to say so (and the paper already leans that
     way in spirit).

All conditions run on the SAME held-out seeds so differences are paired. This costs API
calls, so n is modest by design; it is an ablation, not the large-n anchor (that's the
1000-turn head-to-head). Writes outputs/ablation_results.json + a turn log per condition.
"""
from __future__ import annotations
import json
import time

from . import agent_eval as AE


# The conditions. Each is (label, arch, few_shot_k, sc_samples).
CONDITIONS = [
    ("panel_zero_shot",        "panel",  0, 1),
    ("panel_few_shot_only",    "panel",  6, 1),
    ("panel_self_consistency", "panel",  0, 3),
    ("panel_both",             "panel",  6, 3),
    ("single_zero_shot",       "single", 0, 1),
    ("single_both",            "single", 6, 3),
]


def run(n_scenarios: int = 60, base_seed: int = 20_000, sev_scale: float = 0.5,
        workers: int = 10, train_n: int = 300,
        out_path: str = "outputs/ablation_results.json") -> dict:
    """Run every condition on the SAME n held-out scenarios (seeds disjoint from the
    1000-turn eval at 10_000.. and from train/demos). Returns a compact comparison."""
    conditions = {}
    t0 = time.time()
    for label, arch, k, sc in CONDITIONS:
        print(f"\n=== ablation condition: {label} (arch={arch}, k={k}, N={sc}) ===")
        cfg = AE.EvalConfig(
            n_scenarios=n_scenarios, backend="litellm", sev_scale=sev_scale,
            train_n=train_n, workers=workers, base_seed=base_seed,
            arch=arch, few_shot_k=k, sc_samples=sc,
            log_path=f"outputs/ablation_{label}_turns.jsonl",
            out_path=f"outputs/ablation_{label}_results.json")
        res = AE.run(cfg)
        h = res["head_to_head"]
        conditions[label] = {
            "arch": arch, "few_shot_k": k, "sc_samples": sc,
            "detection_f1": h["agent_detection_f1"],
            "classification_acc": h["agent_classification_acc"],
            "calls_per_window": (1 if arch == "single" else 5) + (sc - 1),
            "ml_classification_acc": h["ml_classification_acc"],
        }
        print(f"    -> det_f1={h['agent_detection_f1']:.3f} "
              f"class_acc={h['agent_classification_acc']:.3f} "
              f"(ML {h['ml_classification_acc']:.3f})")

    out = {
        "n_scenarios": n_scenarios, "base_seed": base_seed, "sev_scale": sev_scale,
        "elapsed_s": round(time.time() - t0, 1),
        "conditions": conditions,
        "readings": {
            "technique_decomposition": (
                "Compare panel_zero_shot vs panel_few_shot_only vs panel_self_consistency "
                "vs panel_both to attribute the classification lift to each lever."),
            "architecture": (
                "Compare panel_* vs single_* at matched levers. If single_both ~= panel_both "
                "at 1/5 the calls, the multi-role decomposition does not earn its cost — "
                "consistent with the 'Illusion of Multi-Agent Advantage' we cite."),
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ablation] wrote {out_path} in {out['elapsed_s']}s")
    return out


if __name__ == "__main__":
    run()
