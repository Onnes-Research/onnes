"""
Train + evaluate the classical ML baseline for CryoOpsBench (the "trained ML"
column that sits between the threshold baseline and the LLM agent).

Reads a dataset produced by scripts/generate_dataset.py (telemetry CSVs +
labels.csv), engineers per-window features over the verified FRTMS schema,
trains a two-stage (detect + classify) model BY SCENARIO (no window leakage),
prints a metrics table in the SAME shape as onnesim/evaluate.score(), and writes
outputs/ml_results.json.

Usage:
    python scripts/train_ml.py --dataset outputs/ml_dataset
    python scripts/train_ml.py --dataset outputs/ml_dataset --folds 5 --seed 0

⚠️ Trained on PLACEHOLDER-physics synthetic data — NOT validated against real
hardware. Numbers describe simulator separability only. See README + PHYSICS_NOTES.md.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import ml_baseline as M

CAVEAT = ("trained on PLACEHOLDER-physics synthetic data — not validated against real hardware")
METRIC_KEYS = ["detect_accuracy", "precision", "recall", "f1", "class_accuracy_on_detected"]


def _print_table(title: str, scores: dict, model_names: list[str]) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    hdr = f"{'metric':30s}" + "".join(f"{n:>14s}" for n in model_names)
    print(hdr)
    print("-" * 76)
    for k in METRIC_KEYS:
        print(f"{k:30s}" + "".join(f"{scores[n][k]:>14}" for n in model_names))
    print("-" * 76)
    print(f"{'n (scenarios scored)':30s}" + "".join(f"{scores[n]['n']:>14}" for n in model_names))
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="outputs/ml_dataset",
                    help="dataset dir with labels.csv + scenario CSVs")
    ap.add_argument("--out", default="outputs/ml_results.json")
    ap.add_argument("--folds", type=int, default=5, help="stratified CV folds (by scenario)")
    ap.add_argument("--test-size", type=float, default=0.25, help="holdout fraction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[ml] loading dataset: {args.dataset}")
    X, y, ids, feat_names = M.build_dataset(args.dataset)
    classes = sorted(set(y.tolist()))
    n_fault = int((y != M.NORMAL).sum())
    print(f"[ml] {X.shape[0]} scenarios x {X.shape[1]} features "
          f"({len(M.FEATURE_CHANNELS)} channels x {len(M.STAT_NAMES)} stats)")
    print(f"[ml] classes ({len(classes)}): {', '.join(classes)}")
    print(f"[ml] normal={X.shape[0] - n_fault}  fault={n_fault}")
    print(f"[ml] split=BY-SCENARIO (1 row = 1 scenario) | folds={args.folds} "
          f"holdout={args.test_size} seed={args.seed}")

    results = M.run_all(X, y, seed=args.seed, test_size=args.test_size, folds=args.folds)
    model_names = list(M.make_models(args.seed).keys())

    _print_table("Cross-validated (out-of-fold, all scenarios) — comparable to agent eval",
                 results["cross_val"], model_names)
    _print_table(f"Held-out test split ({int(args.test_size * 100)}% of scenarios)",
                 results["holdout"], model_names)

    best = results["best_model"]
    cv_best = results["cross_val"][best]
    print(f"\n[ml] BEST model: {best}  ({results['best_selection']})")
    print(f"[ml]   detection    -> accuracy={cv_best['detect_accuracy']} "
          f"precision={cv_best['precision']} recall={cv_best['recall']} f1={cv_best['f1']}")
    print(f"[ml]   classification-> class_accuracy_on_detected={cv_best['class_accuracy_on_detected']}")
    print(f"\n⚠️  {CAVEAT}.")
    print("    Metrics reflect simulator separability, not real-hardware performance.")

    payload = {
        "caveat": CAVEAT,
        "dataset": args.dataset,
        "n_scenarios": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_channels": M.FEATURE_CHANNELS,
        "stat_names": M.STAT_NAMES,
        "classes": classes,
        "split": "by_scenario",
        "folds": args.folds,
        "test_size": args.test_size,
        "seed": args.seed,
        "best_model": best,
        "best_selection": results["best_selection"],
        "cross_val": results["cross_val"],
        "holdout": results["holdout"],
        # Convenience: the single row a paper table would quote (best model, CV).
        "ml_baseline": {"model": best, **{k: cv_best[k] for k in METRIC_KEYS},
                        "n": cv_best["n"], "counts": cv_best["counts"]},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[ml] wrote {args.out}")


if __name__ == "__main__":
    main()
