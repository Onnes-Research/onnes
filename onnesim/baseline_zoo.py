"""
baseline_zoo.py — stronger ML opponents under the SAME seed-addressed protocol.

Every reviewer asked the same thing: a random forest is a fine baseline, but in 2026
it is not enough — show gradient boosting (LightGBM) and a tabular foundation model
(TabPFN-2.5). This script trains the whole zoo on the SAME clean training seeds the
head-to-head uses, evaluates on the SAME held-out eval seeds (sev_scale applied,
identical to the agent's scenarios), and reports macro-F1 + exact-accuracy with
Clopper-Pearson 95% CIs so the comparison is statistically honest.

No LLM calls. Reuses onnesim.benchmark's exact render/feature pipeline so the numbers
are directly comparable to the agent eval. Writes outputs/baseline_zoo.json.
"""
from __future__ import annotations
import json
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from . import benchmark as BM
from . import cryo_engine as CE
from . import ml_baseline as MB
from . import stats as ST
from . import virtual_clone as VC


FAULT_CLASSES = BM.FAULT_CLASSES


def _macro_f1(y_true, y_pred) -> float:
    present = [c for c in FAULT_CLASSES if c in set(map(str, y_true))]
    if not present:
        return 0.0
    _, _, f1, _ = precision_recall_fscore_support(
        list(map(str, y_true)), list(map(str, y_pred)),
        labels=present, average="macro", zero_division=0)
    return float(f1)


def run(train_n: int = 300, eval_n: int = 200, base_seed_eval: int = 10_000,
        sev_scale: float = 0.5, hours: float = 6.0, dt_min: float = 5.0,
        fingerprint_dir: str | None = BM.DEFAULT_FINGERPRINT_DIR,
        out_path: str = "outputs/baseline_zoo.json") -> dict:
    """Train every available model on clean seeds 0..train_n-1, test on the eval seeds
    (10_000..) at sev_scale — the SAME scenarios the agent panel saw. Report per-model
    macro-F1 and exact classification accuracy with 95% CIs."""
    fp = None
    if fingerprint_dir:
        try:
            fp = VC.learn_fingerprint(fingerprint_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[zoo] fingerprint unavailable ({exc}); engine noise only")
    cfg = CE.EngineConfig(fingerprint=fp, realistic=True, imperfections=True)

    # ---- training set: clean, disjoint seeds (same as the head-to-head ML) ----
    print(f"[zoo] building train set (n={train_n}, clean) ...")
    tr_specs = BM.sample_specs(train_n, base_seed=0)
    Xtr = BM.build_X([BM.render(s, cfg, hours, dt_min) for s in tr_specs])
    ytr = np.asarray([s.fault_class for s in tr_specs], dtype=object)

    # ---- eval set: the SAME 200 seeds + sev_scale the agent panel faced --------
    print(f"[zoo] building eval set (n={eval_n}, sev_scale={sev_scale}) ...")
    ev_specs = BM.sample_specs(eval_n, base_seed=base_seed_eval)
    Xte = BM.build_X([BM.render(s, cfg, hours, dt_min, sev_scale=sev_scale) for s in ev_specs])
    yte = np.asarray([s.fault_class for s in ev_specs], dtype=object)

    avail = MB.available_extra_models()
    print(f"[zoo] optional strong models available: {avail}")

    models = MB.make_models(seed=0, include_strong=True)
    models.pop("most_frequent", None)  # not an interesting opponent here

    results = {}
    for name, make in models.items():
        t0 = time.time()
        try:
            est = make()
            est.fit(Xtr, ytr)
            pred = est.predict(Xte)
            acc_k = int(np.sum(pred == yte))
            res = {
                "macro_f1": round(_macro_f1(yte, pred), 3),
                "exact_accuracy": ST.prop_ci(acc_k, eval_n),
                "fit_predict_s": round(time.time() - t0, 1),
            }
            results[name] = res
            ci = res["exact_accuracy"]
            print(f"  {name:16s} macroF1={res['macro_f1']:.3f}  "
                  f"acc={ci['p']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]  ({res['fit_predict_s']}s)")
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": str(exc)}
            print(f"  {name:16s} FAILED: {exc}")

    out = {
        "protocol": {
            "train_n": train_n, "eval_n": eval_n, "base_seed_eval": base_seed_eval,
            "sev_scale": sev_scale, "note": "same seed-addressed scenarios as the agent head-to-head",
        },
        "available_strong_models": avail,
        "models": results,
        "reading": ("All models share the head-to-head's exact train/eval seeds and feature "
                    "pipeline, so these accuracies are directly comparable to the agent panel. "
                    "CIs are Clopper-Pearson 95%."),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[zoo] wrote {out_path}")
    return out


if __name__ == "__main__":
    run()
