"""
eval_real.py — run anomaly detection on REAL labeled telemetry (Server Machine
Dataset) and score it with the SAME onnesim.evaluate.score() used for the fridge
pipeline. Proves the detection METHOD works on real data.

⚠️ SMD is server telemetry, NOT dilution-fridge data. This validates the machinery,
not any cryogenic claim. See docs/DATA_LINKS.md / REAL_DATA_OPTIONS.md.

Download the 3 files first (see docs/DATA_LINKS.md), then:
    python scripts/eval_real.py
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim.real_data import load_smd
from onnesim import evaluate as E

TRAIN = "data/real/smd_machine-1-1_train.txt"
TEST = "data/real/smd_machine-1-1_test.txt"
LABELS = "data/real/smd_machine-1-1_labels.txt"


def main():
    for p in (TRAIN, TEST, LABELS):
        if not os.path.exists(p):
            print(f"[eval_real] missing {p} — see docs/DATA_LINKS.md to download.")
            sys.exit(1)

    from sklearn.ensemble import IsolationForest

    tr = load_smd(TRAIN, window=100, stride=50)
    te = load_smd(TEST, LABELS, window=100, stride=50)
    print(f"[eval_real] REAL Server Machine Dataset (NOT fridge data)")
    print(f"[eval_real] train windows {tr['X_windows'].shape}, "
          f"test windows {te['X_windows'].shape}, {tr['n_channels']} channels")
    print(f"[eval_real] test anomaly windows: {int(te['y'].sum())}/{len(te['y'])}")

    clf = IsolationForest(n_estimators=200, contamination=0.1, random_state=0)
    clf.fit(tr["X_windows"])
    pred = (clf.predict(te["X_windows"]) == -1).astype(int)

    rows = [{"truth_class": "anom" if t else "normal",
             "pred_detected": bool(p), "pred_class": "anom"}
            for t, p in zip(te["y"], pred)]
    s = E.score(rows)

    print("\n=== IsolationForest on REAL labeled telemetry ===")
    for k in ["detect_accuracy", "precision", "recall", "f1"]:
        print(f"  {k:20s} {s[k]}")
    print("  counts", s["counts"])
    print("\nNOTE: server telemetry — validates the METHOD, not any fridge claim.")


if __name__ == "__main__":
    main()
