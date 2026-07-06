"""
Generate a labeled dataset of simulated fridge scenarios (CryoOpsBench v0 seed).

Each scenario = one cooldown with a randomly sampled (possibly 'normal') fault.
Writes one CSV of telemetry per scenario + a labels.csv manifest with ground truth.

Usage:
    python scripts/generate_dataset.py --n 60 --hours 24
"""
from __future__ import annotations
import argparse, os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import FridgeParams, simulate, make_hook, sample_fault, emit, to_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--out", default="outputs/dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fleet-jitter", type=float, default=0.15,
                    help="per-scenario parameter jitter (fleet of virtual fridges)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    duration = args.hours * 3600.0
    rng = np.random.default_rng(args.seed)
    base = FridgeParams()

    manifest_path = os.path.join(args.out, "labels.csv")
    counts = {}
    with open(manifest_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario_id", "csv", "fault_class", "onset_s",
                    "severity", "target_stage", "final_T_mxc"])
        for i in range(args.n):
            spec = sample_fault(rng, duration)
            params = base.perturbed(rng, rel=args.fleet_jitter)
            hook = make_hook(spec)
            sim = simulate(params, duration_s=duration, dt_s=60.0,
                           fault_hook=hook, seed=i)
            cols = emit(sim, spec, seed=i)
            fn = f"scenario_{i:04d}.csv"
            to_csv(cols, os.path.join(args.out, fn))
            w.writerow([i, fn, spec.fault_class, f"{spec.onset_s:.0f}",
                        f"{spec.severity:.3f}", spec.target_stage,
                        f"{sim['T'][-1, -1]:.4g}"])
            counts[spec.fault_class] = counts.get(spec.fault_class, 0) + 1

    print(f"[OnnesSim] wrote {args.n} scenarios to {args.out}/")
    print(f"[OnnesSim] manifest: {manifest_path}")
    print("[OnnesSim] class balance:")
    for k in sorted(counts):
        print(f"    {k:20s} {counts[k]}")


if __name__ == "__main__":
    main()
