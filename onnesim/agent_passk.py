"""
agent_passk.py — pass^k reliability metric (tau-bench standard) for the diagnosis agent.

WHY
---
The paper reports single-run accuracy per seed. The 2026 deployability standard is pass^k:
the probability that the agent is correct on ALL k independent runs of the SAME scenario
(tau-bench; GPT-4o 61% pass@1 -> 25% pass@8 shows why single-run overstates reliability).
For a fridge that runs unattended, consistency IS the requirement, so pass^k is the right
metric. This module runs each scenario k times and reports:

  * pass^k for detection and classification (empirical: fraction of scenarios correct on
    all k runs),
  * an UNBIASED pass^k estimate per scenario from its success count (C(c,k)/C(r,k)),
  * the pass^1 (mean accuracy) baseline for contrast,
  * per-scenario consistency (how many of the k runs agreed with each other),
    which localizes the flaky scenarios — the ones a deployment would distrust.

Backend-agnostic via predict_fn (stub for smoke, panel for live). On the DETERMINISTIC
stub, pass^k == pass^1 by construction (a correctness check on the harness). The interesting
spread only appears with a stochastic backend (temperature>0 or self-consistency sampling).
"""
from __future__ import annotations

import json
import os
from math import comb
from typing import Callable

import numpy as np

from . import cryo_engine as CE
from . import agent_eval as AE
from . import benchmark as BM

Predict = Callable[[dict], "tuple[bool, str]"]
NORMAL = "normal"


def _unbiased_pass_k(n_correct: int, n_runs: int, k: int) -> float:
    """Unbiased P(all k of a random k-subset correct) given n_correct/n_runs observed.

    The HumanEval/tau-bench estimator: 1 - C(n-c, k)/C(n, k) is for pass@k (at least one);
    for pass^k (ALL k correct) the unbiased form is C(c, k)/C(n, k). Returns 0 if c<k.
    """
    if k > n_runs or n_correct < k:
        return 0.0
    return comb(n_correct, k) / comb(n_runs, k)


def run(predict_fn: Predict, n: int = 20, k: int = 5, base_seed: int = 54_000,
        sev_scale: float = 0.5, tag: str = "stub", workers: int | None = None,
        out_path: str | None = None) -> dict:
    """Run each of n scenarios k times; report pass^k for detection + classification.

    Scenarios run concurrently (workers) since each scenario's k runs are an independent
    batch of LLM calls — this is what makes a live pass^k run feasible in bounded time.
    """
    engine_cfg = CE.EngineConfig(realistic=True, imperfections=True)
    specs = BM.sample_specs(n, base_seed=base_seed)

    def _one(spec):
        truth = AE._norm_class(spec.fault_class)
        truth_detected = truth != NORMAL
        det_hits = cls_hits = 0
        preds = []
        for r in range(k):
            scen = CE.Scenario(spec.fault_class, spec.severity * sev_scale, spec.onset_frac)
            cols = CE.simulate(scen, engine_cfg, hours=6.0, dt_min=5.0,
                               seed=spec.seed * 100 + r)
            det, cls = predict_fn(cols)
            det_hits += int(det == truth_detected)
            cls_hits += int(cls == truth)
            preds.append(cls)
        vals, counts = np.unique(preds, return_counts=True)
        return {"seed": spec.seed, "truth": truth, "det_hits": det_hits,
                "cls_hits": cls_hits, "preds": preds,
                "consistency": float(counts.max() / k)}

    from concurrent.futures import ThreadPoolExecutor
    if workers and workers > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_one, specs))
    else:
        rows = [_one(s) for s in specs]

    det_all_correct = cls_all_correct = det_any_correct = cls_any_correct = 0
    pass1_det_sum = pass1_cls_sum = 0.0
    ub_det, ub_cls, flaky = [], [], []
    for row in rows:
        det_hits, cls_hits = row["det_hits"], row["cls_hits"]
        det_all_correct += int(det_hits == k)
        cls_all_correct += int(cls_hits == k)
        det_any_correct += int(det_hits > 0)
        cls_any_correct += int(cls_hits > 0)
        pass1_det_sum += det_hits / k
        pass1_cls_sum += cls_hits / k
        ub_det.append(_unbiased_pass_k(det_hits, k, k))
        ub_cls.append(_unbiased_pass_k(cls_hits, k, k))
        if row["consistency"] < 1.0:
            flaky.append({"seed": row["seed"], "truth": row["truth"],
                          "predictions": row["preds"],
                          "consistency": round(row["consistency"], 2)})

    report = {
        "tag": tag, "n_scenarios": n, "k": k,
        "detection": {
            "pass_1_mean_acc": round(pass1_det_sum / n, 3),
            "pass_k_empirical": round(det_all_correct / n, 3),
            "pass_k_unbiased": round(float(np.mean(ub_det)), 3),
            "pass_at_1_any": round(det_any_correct / n, 3),
        },
        "classification": {
            "pass_1_mean_acc": round(pass1_cls_sum / n, 3),
            "pass_k_empirical": round(cls_all_correct / n, 3),
            "pass_k_unbiased": round(float(np.mean(ub_cls)), 3),
            "pass_at_1_any": round(cls_any_correct / n, 3),
        },
        "n_flaky_scenarios": len(flaky),
        "flaky_examples": flaky[:5],
        "reading": (
            f"pass^{k} = fraction of scenarios correct on ALL {k} runs (tau-bench "
            "deployability metric). The gap pass_1_mean_acc - pass_k_empirical is the "
            "reliability tax single-run accuracy hides. On the deterministic stub the two "
            "coincide (a harness sanity check); a stochastic panel shows the true spread. "
            "Flaky scenarios (consistency<1) are the ones a deployment cannot trust."),
    }
    out_path = out_path or f"outputs/agent_passk_{tag}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    d, c = report["detection"], report["classification"]
    print(f"[passk:{tag}] k={k}  det pass^1={d['pass_1_mean_acc']} pass^k={d['pass_k_empirical']}  "
          f"cls pass^1={c['pass_1_mean_acc']} pass^k={c['pass_k_empirical']}  "
          f"flaky={report['n_flaky_scenarios']}")
    print(f"[passk:{tag}] wrote {out_path}")
    return report


if __name__ == "__main__":
    from . import agent_stress as ST
    run(ST.stub_predict, n=20, k=5, tag="stub")
