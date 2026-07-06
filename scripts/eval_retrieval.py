"""
eval_retrieval.py — does QUERY-CONDITIONED demo retrieval beat a FIXED few-shot block?

The paper's enhanced panel uses a fixed round-robin few-shot block (identical demos for
every scenario). This script tests the new lever (onnesim/retrieval.py): selecting demos
by nearest-neighbour to the query window, with guaranteed contrastive coverage of the
confusable thermal cluster. It runs THREE conditions on the SAME held-out seeds so the
only variable is demo selection:

    zero_shot   : k=0                       (no demos)
    roundrobin  : k=6, few_shot_mode=roundrobin  (the paper's enhanced panel)
    retrieval   : k=6, few_shot_mode=retrieval    (the new lever)

Self-consistency (N) is held fixed across conditions so it cannot confound the comparison.
Outputs go to outputs/retrieval_*.json / .jsonl (paper artifacts untouched), and the paired
McNemar lift roundrobin->retrieval is computed on the shared scenarios via onnesim.stats.

Usage:
    .venv/bin/python scripts/eval_retrieval.py --n 200            # full
    .venv/bin/python scripts/eval_retrieval.py --n 40 --sc 1      # quick, no voting
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import agent_eval as AE
from onnesim import stats as ST


def run_condition(name: str, n: int, k: int, mode: str, sc: int, workers: int,
                  backend: str = "litellm") -> dict:
    cfg = AE.EvalConfig(
        n_scenarios=n, backend=backend, sev_scale=0.5, base_seed=10_000,
        few_shot_k=k, few_shot_mode=mode, sc_samples=sc, workers=workers, train_n=300,
        log_path=f"outputs/retrieval_{name}_turns.jsonl",
        out_path=f"outputs/retrieval_{name}_results.json")
    print(f"\n{'='*72}\n[retrieval] condition '{name}'  (k={k}, mode={mode}, N={sc}, n={n}, backend={backend})\n{'='*72}")
    res = AE.run(cfg)
    h = res["head_to_head"]
    return {"label": name, "n": res["n_scenarios"],
            "det_f1": h["agent_detection_f1"], "cls": h["agent_classification_acc"],
            "ml_cls": h["ml_classification_acc"],
            "log": cfg.log_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sc", type=int, default=3, help="self-consistency votes (held fixed across conditions)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backend", default="litellm", help="litellm (Claude) or gemini")
    ap.add_argument("--skip-zero", action="store_true", help="reuse existing zero-shot anchor")
    args = ap.parse_args()

    rows = []
    if not args.skip_zero:
        rows.append(run_condition("zero_shot", args.n, 0, "roundrobin", 1, args.workers, args.backend))
    rows.append(run_condition("roundrobin", args.n, 6, "roundrobin", args.sc, args.workers, args.backend))
    rows.append(run_condition("retrieval", args.n, 6, "retrieval", args.sc, args.workers, args.backend))

    # paired McNemar: does retrieval fix scenarios roundrobin got wrong (on shared seeds)?
    lift = None
    rr = "outputs/retrieval_roundrobin_turns.jsonl"
    rt = "outputs/retrieval_retrieval_turns.jsonl"
    if os.path.exists(rr) and os.path.exists(rt):
        try:
            lift = ST.paired_lift(rr, rt)   # b: retrieval right & roundrobin wrong (the lift)
            json.dump(lift, open("outputs/retrieval_lift_stats.json", "w"), indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"[retrieval] paired lift failed: {exc}")

    print(f"\n{'='*84}\nDEMO-SELECTION ABLATION — same seeds, only demo selection differs\n{'='*84}")
    print(f"{'condition':28s} {'n':>4s} {'det F1':>8s} {'class acc':>10s}")
    print("-" * 84)
    for r in rows:
        print(f"{r['label']:28s} {r['n']:>4d} {r['det_f1']:>8.3f} {r['cls']:>10.3f}")
    if rows:
        print("-" * 84)
        print(f"{'Supervised RF (shared)':28s} {rows[-1]['n']:>4d} {'':>8s} {rows[-1]['ml_cls']:>10.3f}")
    if lift:
        mc = lift["mcnemar_lift"]
        print(f"\n  paired lift roundrobin -> retrieval (n={lift['n_shared_scenarios']}): "
              f"roundrobin acc={lift['zero_shot']['p']:.3f}, retrieval acc={lift['enhanced']['p']:.3f}")
        print(f"  McNemar p={mc['p_value']:.3g}  (retrieval fixed {mc['improved_by_enhancement']}, "
              f"regressed {mc['regressed']})")
    print("=" * 84)

    summary = {"n": args.n, "sc": args.sc, "conditions": rows, "retrieval_lift": lift}
    with open("outputs/retrieval_ablation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[retrieval] wrote outputs/retrieval_ablation.json")


if __name__ == "__main__":
    main()
