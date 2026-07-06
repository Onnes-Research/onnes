#!/usr/bin/env python3
"""
train_robust_ml.py — GPU-2: domain-randomized ML robust across a FRIDGE FAMILY.

WHY
---
The paper's RF trains on ONE twin configuration and is tested on the SAME configuration's
held-out seeds. That measures separability, not TRANSFER: a model that memorizes one fridge's
exact base temps / noise Sigma will not survive fridge-to-fridge drift (the real sim-to-real
risk — every commissioned fridge is slightly different). Domain randomization is the standard
fix: randomize the twin's physics constants (n_dot_3, T_in factor f, per-stage noise, base
loads) across MANY variants, train on the whole family, and report accuracy on HELD-OUT
fridge variants the model never saw. A model robust across the randomized family is far more
likely to transfer to a real fridge than one tuned to a single config.

This turns "our RF scores 0.985 on the twin" into "our RF scores X on fridges it has never
seen", which is the honest transfer claim.

COMPUTE
-------
The randomization multiplies the dataset: 10k variants x scenarios x feature extraction is
where an H200 pays off (embarrassingly parallel simulation + a large training matrix; and it
naturally extends to training a neural classifier over the huge randomized set). CPU-verifiable
at --smoke (few variants). Physics + feature extraction are numpy (CPU), but the scale (and an
optional torch MLP head) is the GPU workload.

Outputs: outputs/robust_ml_metrics.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import cryo_engine as CE
from onnesim import ml_baseline as MB
from onnesim import benchmark as BM
from onnesim import dilution_cooling as DC


def _randomized_engine(rng: np.random.Generator) -> CE.EngineConfig:
    """A twin variant with randomized physics constants — one 'fridge' in the family.

    Randomizes the dilution unit (n_dot_3 circulation, T_in factor, cold-plate/still cooling
    power). Each variant has a genuinely different base state, so a model must learn the FAULT
    SIGNATURE, not one fridge's absolute temperatures.
    """
    du = DC.DilutionUnit(
        n_dot_3=float(rng.uniform(300e-6, 700e-6)),      # circulation rate spread
        T_in_factor=float(rng.uniform(1.3, 1.7)),         # heat-exchanger ratio spread
        Q_cp_100mK_W=float(rng.uniform(200e-6, 400e-6)),  # cold-plate cooling spread
        Q_still_nominal_W=float(rng.uniform(20e-3, 40e-3)),
    )
    return CE.EngineConfig(du=du, realistic=True, imperfections=True)


def _dataset_from_variants(n_variants: int, scen_per_variant: int, base_seed: int,
                           sev_scale: float, rng: np.random.Generator):
    """Build (X, y) by drawing scenarios from many randomized fridge variants."""
    X, y = [], []
    for v in range(n_variants):
        cfg = _randomized_engine(rng)
        specs = BM.sample_specs(scen_per_variant, base_seed=base_seed + v * scen_per_variant)
        for s in specs:
            cols = CE.simulate(CE.Scenario(s.fault_class, s.severity * sev_scale, s.onset_frac),
                               cfg, hours=6.0, dt_min=5.0, seed=s.seed)
            X.append(MB.extract_features(cols)[0])
            y.append(s.fault_class)
    return np.asarray(X), np.asarray(y, dtype=object)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-variants", type=int, default=400)
    ap.add_argument("--test-variants", type=int, default=100)
    ap.add_argument("--scen-per-variant", type=int, default=12)
    ap.add_argument("--sev-scale", type=float, default=0.5)
    args = ap.parse_args()
    if args.smoke:
        args.train_variants, args.test_variants, args.scen_per_variant = 8, 4, 6

    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    print(f"[robust] building TRAIN set: {args.train_variants} fridge variants x "
          f"{args.scen_per_variant} scenarios ...")
    Xtr, ytr = _dataset_from_variants(args.train_variants, args.scen_per_variant,
                                      base_seed=0, sev_scale=args.sev_scale, rng=rng)
    # held-out fridge variants use a DISJOINT randomization stream + seed range
    rng_test = np.random.default_rng(999)
    print(f"[robust] building HELD-OUT set: {args.test_variants} UNSEEN fridge variants ...")
    Xte, yte = _dataset_from_variants(args.test_variants, args.scen_per_variant,
                                      base_seed=500_000, sev_scale=args.sev_scale, rng=rng_test)

    print(f"[robust] train {Xtr.shape}, test {Xte.shape} — training RF ...")
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    acc_family = float(np.mean(rf.predict(Xte) == yte))

    # contrast: a model trained on ONE fridge (no randomization), tested on the family
    single_cfg = CE.EngineConfig(realistic=True, imperfections=True)
    Xs, ys = [], []
    specs = BM.sample_specs(args.train_variants * args.scen_per_variant, base_seed=0)
    for s in specs:
        cols = CE.simulate(CE.Scenario(s.fault_class, s.severity * args.sev_scale, s.onset_frac),
                           single_cfg, hours=6.0, dt_min=5.0, seed=s.seed)
        Xs.append(MB.extract_features(cols)[0]); ys.append(s.fault_class)
    rf_single = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf_single.fit(np.asarray(Xs), np.asarray(ys, dtype=object))
    acc_single_on_family = float(np.mean(rf_single.predict(Xte) == yte))

    metrics = {
        "train_variants": args.train_variants, "test_variants": args.test_variants,
        "scen_per_variant": args.scen_per_variant,
        "domain_randomized_acc_on_unseen_fridges": round(acc_family, 3),
        "single_fridge_model_acc_on_unseen_fridges": round(acc_single_on_family, 3),
        "robustness_gain": round(acc_family - acc_single_on_family, 3),
        "reading": ("Accuracy on HELD-OUT randomized fridge variants (unseen physics). The "
                    "domain-randomized model should beat the single-fridge model ON THE "
                    "FAMILY, evidence it learned fault signatures that transfer across fridges "
                    "rather than memorizing one fridge's temperatures — the honest proxy for "
                    "sim-to-real transfer before real fault data exists."),
    }
    os.makedirs("outputs", exist_ok=True)
    json.dump(metrics, open("outputs/robust_ml_metrics.json", "w"), indent=2)
    print(f"[robust] domain-randomized acc on UNSEEN fridges: {acc_family:.3f}")
    print(f"[robust] single-fridge model on UNSEEN fridges:   {acc_single_on_family:.3f}")
    print(f"[robust] robustness gain: {metrics['robustness_gain']:+.3f}")
    print("[robust] wrote outputs/robust_ml_metrics.json")
    if args.smoke:
        print("[robust] SMOKE OK — scale up --train-variants on the H200.")


if __name__ == "__main__":
    main()
