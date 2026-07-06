"""
Run a single simulated cooldown and save a CSV + a PNG of the stage curves.

Usage:
    python scripts/run_cooldown.py [--fault helium_leak] [--hours 36]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import FridgeParams, simulate, FaultSpec, make_hook, emit, to_csv
from onnesim import faults as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="normal", choices=F.FAULT_CLASSES)
    ap.add_argument("--hours", type=float, default=36.0)
    ap.add_argument("--severity", type=float, default=0.8)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    duration = args.hours * 3600.0

    spec = FaultSpec(
        fault_class=args.fault,
        onset_s=0.45 * duration if args.fault != "normal" else duration * 2,
        severity=args.severity if args.fault != "normal" else 0.0,
        target_stage=F.MXC,
        flow_scale=(1.0 - 0.6 * args.severity) if args.fault == "blocked_impedance" else 1.0,
    )
    hook = make_hook(spec)

    sim = simulate(FridgeParams(), duration_s=duration, dt_s=30.0, fault_hook=hook)
    cols = emit(sim, spec, seed=1)

    csv_path = os.path.join(args.out, "cooldown.csv")
    to_csv(cols, csv_path)

    final = sim["T"][-1]
    print(f"[OnnesSim] fault={args.fault} severity={spec.severity}")
    print(f"[OnnesSim] final stage temps [K]:")
    for name, tv in zip(sim["stages"], final):
        print(f"    {name:10s} {tv:10.4g}")
    print(f"[OnnesSim] wrote {csv_path}  ({sim['T'].shape[0]} rows)")

    # Plot (best-effort; skip if matplotlib missing).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t_h = sim["t_s"] / 3600.0
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, name in enumerate(sim["stages"]):
            ax.semilogy(t_h, sim["T"][:, i], label=name)
        if spec.severity > 0:
            ax.axvline(spec.onset_s / 3600.0, ls="--", c="k", alpha=0.5,
                       label=f"fault onset ({args.fault})")
        ax.set_xlabel("time [h]"); ax.set_ylabel("stage temperature [K] (log)")
        ax.set_title("OnnesSim v0 cooldown  ⚠️ phenomenological / PLACEHOLDER constants")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        png = os.path.join(args.out, "cooldown.png")
        fig.tight_layout(); fig.savefig(png, dpi=120)
        print(f"[OnnesSim] wrote {png}")
    except Exception as e:
        print(f"[OnnesSim] plot skipped: {e}")


if __name__ == "__main__":
    main()
