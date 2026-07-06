"""
run_benchmark.py — run the HONEST CryoOpsBench harness and print the full report.

Generates its OWN scenarios from the unchanged cryo_engine, holds out BY SEED,
compares a trivial floor + FRTMS threshold rules + logreg + random forest +
gradient boosting, prints per-class precision/recall, the confusion matrix, the
biggest confusions, and a realism stress test — then writes the whole thing to
outputs/benchmark_results.json.

Usage:
    .venv/bin/python scripts/run_benchmark.py [--n 360] [--hours 6] [--dt-min 5]
        [--no-fingerprint] [--out outputs/benchmark_results.json]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import benchmark as B


def _bar(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def _fmt_pct_row(name: str, m: dict) -> str:
    s = m["score"]
    return (f"  {name:16s}  detF1={s['f1']:.3f}  detAcc={s['detect_accuracy']:.3f}  "
            f"P={s['precision']:.3f}  R={s['recall']:.3f}  "
            f"macroF1={m['macro_f1']:.3f}  grpF1={m['grouped_macro_f1']:.3f}  "
            f"acc={m['overall_multiclass_accuracy']:.3f}")


def print_report(res: dict) -> None:
    cfg = res["config"]
    _bar("CryoOpsBench — HONEST harness (generated from cryo_engine, held out BY SEED)")
    print(f"  scenarios: n={cfg['n']}  train={cfg['n_train']}  test={cfg['n_test']}  "
          f"window={cfg['hours']}h @ {cfg['dt_min']}min")
    print(f"  real-fridge noise fingerprint: {cfg['fingerprint']}"
          + (f"  ({cfg['fingerprint_dir']})" if cfg["fingerprint"] else "  (engine built-in noise)"))
    print(f"  holdout: {cfg['holdout']}")
    print(f"  class balance: {res['class_balance']}")

    _bar("MODEL COMPARISON  (detF1 = fault-vs-normal; macroF1 = 6-class; "
         "grpF1 = ambiguous thermal cluster forgiven)")
    for name, m in res["models"].items():
        print(_fmt_pct_row(name, m))
    print(f"\n  best learned model: {res['best_learned_model']}")

    best = res["best_learned_model"]
    bm = res["models"][best]
    _bar(f"PER-CLASS precision / recall / f1  —  best model: {best}")
    print(f"  {'class':22s} {'prec':>6s} {'recall':>7s} {'f1':>6s} {'support':>8s}")
    for c, d in bm["per_class"].items():
        print(f"  {c:22s} {d['precision']:6.3f} {d['recall']:7.3f} {d['f1']:6.3f} {d['support']:8d}")

    _bar(f"CONFUSION MATRIX  —  best model: {best}   (rows = truth, cols = predicted)")
    labels = bm["confusion_labels"]
    short = [l[:9] for l in labels]
    print(" " * 22 + "".join(f"{s:>11s}" for s in short))
    for i, row in enumerate(bm["confusion_matrix"]):
        print(f"  {labels[i]:20s}" + "".join(f"{v:>11d}" for v in row))
    print("\n  biggest confusions (truth -> pred) on CLEAN default data:")
    clean_conf = res["top_confusions"][best]
    if clean_conf:
        for c in clean_conf:
            print(f"    {c['truth']:22s} -> {c['pred']:22s}  x{c['count']}")
    else:
        print("    (none — matrix is perfectly diagonal; the faults are trivially")
        print("     separable at default settings. The REAL confusion story is below.)")

    cus = res["confusion_under_stress"]
    _bar(f"CONFUSION MATRIX UNDER STRESS  —  {best} @ noise={cus['noise_level']}   "
         f"(macroF1={cus['macro_f1']:.3f}) — THE REAL STORY")
    print(" " * 22 + "".join(f"{s:>11s}" for s in short))
    for i, row in enumerate(cus["confusion_matrix"]):
        print(f"  {labels[i]:20s}" + "".join(f"{v:>11d}" for v in row))
    print("\n  biggest confusions (truth -> pred) under stress:")
    for c in res["top_confusions_under_stress"]:
        print(f"    {c['truth']:22s} -> {c['pred']:22s}  x{c['count']}")

    _bar("REALISM STRESS TEST  (train clean; test degraded — does accuracy fall?)")
    for axis, rows in res["stress_test"].items():
        print(f"\n  -- {axis} --")
        model_names = [k for k in rows[0].keys() if k != "level"]
        header = "    level  " + "  ".join(f"{mn+'(macroF1/acc)':>26s}" for mn in model_names)
        print(header)
        for r in rows:
            cells = []
            for mn in model_names:
                cells.append(f"{r[mn]['macro_f1']:.3f}/{r[mn]['overall_acc']:.3f}".rjust(26))
            print(f"    {r['level']:<6}" + "  ".join(cells))

    _bar("LABEL-NOISE ROBUSTNESS  (flip x% of TRAIN labels; if test barely moves -> too easy)")
    for frac, d in res["label_noise_robustness"].items():
        print(f"    train label noise {frac}:  detF1={d['detection_f1']:.3f}  macroF1={d['macro_f1']:.3f}")

    v = res["verdict"]
    _bar("HONEST VERDICT")
    print(f"  best model                       : {v['best_model']}")
    print(f"  clean detection F1               : {v['clean_detection_f1']:.3f}")
    print(f"  clean macro F1 (6-class)         : {v['clean_macro_f1']:.3f}")
    print(f"  macro F1 @ heavy noise (0.20)    : {v['macro_f1_heavy_noise_0.20']:.3f}")
    print(f"  macro F1 @ short window (0.10)   : {v['macro_f1_short_window_0.10']:.3f}")
    print(f"  macro F1 @ low severity (0.15)   : {v['macro_f1_low_severity_0.15']:.3f}")
    print(f"  WORST-CASE macro F1 under stress : {v['worst_case_macro_f1_under_stress']:.3f}")
    print(f"  target band                      : {v['target_band']}")
    print(f"\n  >>> {v['label']} <<<")

    if v.get("default_setting_is_trivial"):
        print("\n  What would make it realistically hard (ties to docs/FINDINGS.md #2-4 and the")
        print("  parallel realistic_faults work):")
        print("   1. magnet_quench as a fast, large, transient excursion with a dQ/dt signature")
        print("      (FINDINGS #2) — currently it barely warms 4K, so it's trivially separable.")
        print("   2. Overlap the thermal faults on purpose (helium_leak / wiring_heat_ingress /")
        print("      quench share 'cold stages warm slowly, flow flat', FINDINGS #3) and force the")
        print("      model to use the distinguishing channels (pressure/flow transients).")
        print("   3. Add sensor dropouts, partial faults, and MULTI-fault scenarios.")
        print("   4. Raise the noise floor toward the real fridge's and shorten default windows.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=360, help="total scenarios (train+test)")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--dt-min", type=float, default=5.0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fingerprint", action="store_true",
                    help="use the engine's built-in noise instead of the real BlueFors fingerprint")
    ap.add_argument("--out", default="outputs/benchmark_results.json")
    args = ap.parse_args()

    fp_dir = None if args.no_fingerprint else B.DEFAULT_FINGERPRINT_DIR
    print(f"[benchmark] generating {args.n} scenarios from cryo_engine "
          f"(fingerprint={'off' if args.no_fingerprint else fp_dir}) ...")
    res = B.run(n=args.n, hours=args.hours, dt_min=args.dt_min, test_frac=args.test_frac,
                base_seed=args.base_seed, fingerprint_dir=fp_dir, seed=args.seed)
    print_report(res)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(B._jsonify(res), fh, indent=2)
    print(f"\n[benchmark] full results written to {args.out}")


if __name__ == "__main__":
    main()
