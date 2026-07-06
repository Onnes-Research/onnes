"""
eval_gemini.py — reproduce the paper's agent head-to-head on GEMINI 3.1 PRO.

The paper's numbers were produced with Claude Opus 4.8 (backend="litellm"). This script
runs the IDENTICAL evaluation protocol on Gemini 3.1 Pro via the Google GenAI SDK
(backend="gemini", multi_agent._ask_gemini), so the ONLY variable is the model:
  * same seed-addressed scenarios (base_seed 10_000, the eval range),
  * same realism stressor (sev_scale 0.5),
  * same 5-role panel, same summary features, same scorer,
  * same supervised RF opponent (trained on the same clean seeds).

It runs two conditions, mirroring the paper:
  1. zero-shot   (few_shot_k=0, sc_samples=1)
  2. enhanced    (few_shot_k=6, sc_samples=3)   — the two ICL levers

Gemini artifacts go to outputs/gemini_*.json / .jsonl so the Claude artifacts the paper
cites are NEVER overwritten. At the end it prints a Claude-vs-Gemini scorecard, reading
the Claude numbers back from the existing outputs/ JSON.

Usage (key via env, never hard-coded):
    GEMINI_API_KEY=... .venv/bin/python scripts/eval_gemini.py --n 200
    GEMINI_API_KEY=... .venv/bin/python scripts/eval_gemini.py --n 40 --zero-only   # quick
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import agent_eval as AE


def _load(path: str) -> dict | None:
    return json.load(open(path)) if os.path.exists(path) else None


def _claude_row(path: str, label: str) -> dict | None:
    d = _load(path)
    if not d:
        return None
    h = d["head_to_head"]
    return {"label": label, "n": d.get("n_scenarios"),
            "det_f1": h["agent_detection_f1"], "cls": h["agent_classification_acc"],
            "ml_det_f1": h["ml_detection_f1"], "ml_cls": h["ml_classification_acc"]}


def run_condition(name: str, n: int, k: int, sc: int, workers: int) -> dict:
    cfg = AE.EvalConfig(
        n_scenarios=n, backend="gemini", sev_scale=0.5, base_seed=10_000,
        few_shot_k=k, sc_samples=sc, workers=workers, train_n=300,
        log_path=f"outputs/gemini_{name}_turns.jsonl",
        out_path=f"outputs/gemini_{name}_results.json",
    )
    print(f"\n{'='*72}\n[gemini] condition '{name}'  (k={k}, N={sc}, n={n})\n{'='*72}")
    res = AE.run(cfg)
    h = res["head_to_head"]
    return {"label": name, "n": res["n_scenarios"],
            "det_f1": h["agent_detection_f1"], "cls": h["agent_classification_acc"],
            "ml_det_f1": h["ml_detection_f1"], "ml_cls": h["ml_classification_acc"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="scenarios (paper uses 200)")
    ap.add_argument("--workers", type=int, default=4, help="parallel scenarios (Gemini RPM-limited)")
    ap.add_argument("--zero-only", action="store_true", help="run only the zero-shot condition")
    ap.add_argument("--enhanced-only", action="store_true", help="run only the enhanced condition")
    args = ap.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("[gemini] set GEMINI_API_KEY in the environment first."); sys.exit(1)

    model = os.environ.get("ONNES_GEMINI_MODEL", "gemini-3.1-pro-preview")
    print(f"[gemini] model = {model}   (Google GenAI SDK, backend='gemini')")
    print(f"[gemini] n={args.n}  sev_scale=0.5  base_seed=10000 (same seeds as the Claude paper run)")

    rows = []
    if not args.enhanced_only:
        rows.append(run_condition("zero_shot", args.n, 0, 1, args.workers))
    if not args.zero_only:
        rows.append(run_condition("enhanced", args.n, 6, 3, args.workers))

    # ---- Claude-vs-Gemini scorecard -----------------------------------------
    claude_zero = _claude_row("outputs/agent_eval_results.json", "Claude Opus 4.8 zero-shot")
    claude_enh = _claude_row("outputs/agent_eval_fewshot_n200_results.json", "Claude Opus 4.8 enhanced")

    print(f"\n{'='*84}\nBACKEND COMPARISON — same benchmark, same seeds, only the LLM differs\n{'='*84}")
    print(f"{'system':38s} {'n':>4s} {'det F1':>8s} {'class acc':>10s}")
    print("-" * 84)
    for r in (claude_zero, claude_enh):
        if r:
            print(f"{r['label']:38s} {r['n']:>4d} {r['det_f1']:>8.3f} {r['cls']:>10.3f}")
    gmodel = model.replace("-preview", "")
    for r in rows:
        lab = f"Gemini 3.1 Pro {r['label']}"
        print(f"{lab:38s} {r['n']:>4d} {r['det_f1']:>8.3f} {r['cls']:>10.3f}")
    # the supervised opponent is identical in both runs (same seeds/features)
    if rows:
        print("-" * 84)
        print(f"{'Supervised RF (shared opponent)':38s} {rows[0]['n']:>4d} "
              f"{rows[0]['ml_det_f1']:>8.3f} {rows[0]['ml_cls']:>10.3f}")
    print("=" * 84)

    summary = {"model": model, "n": args.n, "gemini": rows,
               "claude_zero_shot": claude_zero, "claude_enhanced": claude_enh}
    with open("outputs/gemini_vs_claude.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[gemini] wrote outputs/gemini_vs_claude.json")


if __name__ == "__main__":
    main()
