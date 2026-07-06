"""
train_surrogate.py — train & save the real ML surrogate of the Cryowala physics.

The surrogate emulates onnesim.cryowala_physics.all_stage_temps (a BSD-3
CryowalaCore-calibrated, real-fridge-fit heat-load->temperature map) with a torch
MLP. After the physics domain fix (train only on physically-valid samples), it
reaches ~0.01% mean relative error on a held-out set.

USAGE:
    python scripts/train_surrogate.py                 # CPU, ~2 min
    ONNES_DEVICE=cuda python scripts/train_surrogate.py   # GPU (H200) for fleet-scale

Saves the model + normalization stats to outputs/surrogate.pt and metrics to
outputs/surrogate_metrics.json.

⚠️ HONEST SCOPE: this is a faithful, accurate emulator of the Cryowala MODEL, which
is calibrated to its authors' fridge — NOT validated against your hardware. Low
error here means "the NN learned the physics map well," not "matches your fridge."
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import ml_surrogate as S


def main():
    try:
        import torch
    except ImportError:
        print("[train] torch not installed. `pip install torch` (see requirements.txt).")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    print("[train] training surrogate on the real (physically-filtered) Cryowala map...")
    r = S.train_surrogate(verbose=True)

    labels = ["50K", "4K", "Still", "CP", "MXC"]
    metrics = {
        "device": r["device"],
        "n_train": r["n_train"], "n_val": r["n_val"],
        "mean_rel_err_pct": round(r["mean_rel_err"] * 100, 4),
        "per_stage_median_pct": {l: round(float(m) * 100, 4)
                                 for l, m in zip(labels, r["median_rel_err_per_stage"])},
        "per_stage_p90_pct": {l: round(float(m) * 100, 4)
                              for l, m in zip(labels, r["p90_rel_err_per_stage"])},
    }
    print("\n[train] held-out accuracy (emulating the REAL Cryowala map):")
    print(f"  mean relative error: {metrics['mean_rel_err_pct']} %")
    for l in labels:
        print(f"  {l:5s} median {metrics['per_stage_median_pct'][l]:.4f}%  "
              f"p90 {metrics['per_stage_p90_pct'][l]:.4f}%")

    # save model + norm stats
    xm, xs, ym, ys = r["norm"]
    torch.save({"state_dict": r["model"].state_dict(),
                "norm": {"xm": xm.tolist(), "xs": xs.tolist(),
                         "ym": ym.tolist(), "ys": ys.tolist()}},
               "outputs/surrogate.pt")
    with open("outputs/surrogate_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("\n[train] saved outputs/surrogate.pt + outputs/surrogate_metrics.json")
    print("[train] NOTE: emulates the Cryowala MODEL (its authors' fridge), not your hardware.")


if __name__ == "__main__":
    main()
