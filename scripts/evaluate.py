"""
Evaluate CryoAgent vs the FRTMS-style threshold baseline on CryoOpsBench scenarios.

Reads the dataset produced by generate_dataset.py (telemetry CSVs + labels.csv),
runs both systems on each scenario, and prints a comparison table.

Usage:
    python scripts/evaluate.py --dataset outputs/dataset --backend auto [--limit N]

Backends:
    auto   : use Claude if ANTHROPIC_API_KEY + anthropic SDK are present, else stub
    claude : force Claude (errors if unavailable)
    stub   : force the transparent rule-based fallback (NOT a model)
"""
from __future__ import annotations
import argparse, csv, os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import agent as A
from onnesim import evaluate as E


def load_cols(path: str) -> dict:
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {name: data[:, i] for i, name in enumerate(header)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--backend", default="auto", choices=["auto", "litellm", "claude", "stub"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all scenarios")
    ap.add_argument("--out", default="outputs/eval_results.json")
    args = ap.parse_args()

    labels_path = os.path.join(args.dataset, "labels.csv")
    if not os.path.exists(labels_path):
        print(f"[eval] no labels.csv in {args.dataset}. Run generate_dataset.py first.")
        sys.exit(1)

    with open(labels_path) as fh:
        manifest = list(csv.DictReader(fh))
    if args.limit:
        manifest = manifest[: args.limit]

    agent_rows, base_rows, per_scenario = [], [], []
    backend_used = None
    for i, row in enumerate(manifest):
        cols = load_cols(os.path.join(args.dataset, row["csv"]))
        truth = row["fault_class"]

        verdict = A.run_agent(cols, backend=args.backend)
        backend_used = verdict.backend
        base = E.threshold_baseline(cols)

        agent_rows.append({"truth_class": truth,
                           "pred_detected": verdict.fault_detected,
                           "pred_class": verdict.fault_class})
        base_rows.append({"truth_class": truth,
                          "pred_detected": base["fault_detected"],
                          "pred_class": base["fault_class"]})
        per_scenario.append({
            "scenario": row["csv"], "truth": truth,
            "agent": [verdict.fault_detected, verdict.fault_class],
            "baseline": [base["fault_detected"], base["fault_class"]],
        })
        print(f"  [{i+1}/{len(manifest)}] {row['csv']:20s} truth={truth:18s} "
              f"agent={verdict.fault_class:18s} base={base['fault_class']}")

    agent_score = E.score(agent_rows)
    base_score = E.score(base_rows)

    print("\n" + "=" * 68)
    print(f"CryoOpsBench v0 results   (agent backend: {backend_used})")
    print("=" * 68)
    hdr = f"{'metric':32s} {'threshold_baseline':>18s} {'CryoAgent':>14s}"
    print(hdr); print("-" * 68)
    for k in ["detect_accuracy", "precision", "recall", "f1",
              "class_accuracy_on_detected"]:
        print(f"{k:32s} {base_score[k]:>18} {agent_score[k]:>14}")
    print("=" * 68)
    if backend_used == "stub":
        print("⚠️  backend=stub is a rule-based fallback, NOT a language model.")
        print("    Set ANTHROPIC_API_KEY and install `anthropic` for real Claude results.")

    with open(args.out, "w") as fh:
        json.dump({"backend": backend_used, "agent": agent_score,
                   "baseline": base_score, "per_scenario": per_scenario}, fh, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
