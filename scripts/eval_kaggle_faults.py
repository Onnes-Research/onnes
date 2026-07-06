"""
eval_kaggle_faults.py — real FAULT detection/classification on labeled refrigeration data.

Dataset (VERIFIED, downloaded, Apache-2.0): Kaggle "Simulated Refrigerator Fault
Diagnosis Dataset" (samoilovmikhail). 21-column time series with fault labels
NORMAL / COND_FOUL_MILD / COND_FOUL_SEVERE / EVAP_FAN_DEG across ~348 runs.

⚠️ HONESTY: this is SIMULATED COMMERCIAL refrigeration, NOT a millikelvin dilution
fridge. It has the labeled FAULTS that the real Leeds cryo logs lack, so it proves
the fault detection+classification PIPELINE on real labeled data. It is NOT a cryo
result. Cryo-specific fault performance still needs real fridge fault data (outreach).

Split is BY RUN (run_id) so no window leaks across train/test.
Usage: python scripts/eval_kaggle_faults.py [--max-runs N]
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import evaluate as E

CSV = "data/real/kaggle_fridge/fridge_fault_timeseries_dataset.csv"
# physical feature columns (exclude ids/labels/time)
FEATURES = ["T_amb", "T_set", "T_cab", "T_evap_sat", "T_cond_sat", "P_suc_bar",
            "P_dis_bar", "N_comp_Hz", "SH_K", "P_comp_W", "Q_evap_W", "COP",
            "frost_level", "T_cab_meas", "valve_open"]


def load_runs(max_runs: int = 0):
    """Group rows by run_id; one feature vector (mean/std/min/max/slope per channel)
    + one label per run. Returns X, y_class, run_ids."""
    with open(CSV) as fh:
        r = csv.reader(fh); hdr = next(r)
        idx = {h: i for i, h in enumerate(hdr)}
        runs = {}
        for row in r:
            rid = row[idx["run_id"]]
            runs.setdefault(rid, {"rows": [], "fault": row[idx["fault"]]})
            runs[rid]["rows"].append(row)
            if max_runs and len(runs) > max_runs and rid not in runs:
                break
    X, y, ids = [], [], []
    for rid, d in runs.items():
        arr = np.array([[float(row[idx[c]]) for c in FEATURES] for row in d["rows"]])
        t = np.linspace(0, 1, len(arr))
        feat = []
        for j in range(arr.shape[1]):
            col = arr[:, j]
            slope = float(np.polyfit(t, col, 1)[0]) if np.ptp(col) > 0 else 0.0
            feat += [col.mean(), col.std(), col.min(), col.max(), slope]
        X.append(feat); y.append(d["fault"]); ids.append(rid)
    return np.array(X), np.array(y, dtype=object), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(CSV):
        print(f"[kaggle] missing {CSV} — see docs/DATA_LINKS.md."); sys.exit(1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("[kaggle] loading real labeled refrigeration-fault runs (BY RUN)...")
    X, y, ids = load_runs(args.max_runs)
    classes, counts = np.unique(y, return_counts=True)
    print(f"[kaggle] {len(X)} runs | classes: {dict(zip(classes, counts.tolist()))}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    # detection (fault-vs-NORMAL) + classification, via evaluate.score()
    rows = [{"truth_class": str(t), "pred_detected": (p != "NORMAL"),
             "pred_class": str(p)} for t, p in zip(yte, pred)]
    # relabel truth NORMAL so score()'s is_fault logic works
    rows = [{"truth_class": ("normal" if t == "NORMAL" else str(t)),
             "pred_detected": bool(p != "NORMAL"), "pred_class": str(p)}
            for t, p in zip(yte, pred)]
    s = E.score(rows)
    overall = float(np.mean(pred == yte))

    print("\n=== RandomForest on REAL labeled refrigeration faults (by-run split) ===")
    print(f"  overall multiclass accuracy: {overall:.3f}")
    for k in ["detect_accuracy", "precision", "recall", "f1", "class_accuracy_on_detected"]:
        print(f"  {k:28s} {s[k]}")
    print("\nNOTE: SIMULATED COMMERCIAL refrigeration, not mK cryo. Proves the")
    print("      fault detect+classify PIPELINE on real labeled data; not a cryo claim.")


if __name__ == "__main__":
    main()
