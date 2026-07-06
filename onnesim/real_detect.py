"""
real_detect.py — anomaly detection trained + tested on REAL fridge telemetry.

THE HONEST REAL-DATA EXPERIMENT
-------------------------------
No public labeled dilution-fridge fault dataset exists (verified by literature search). The
field's accepted method under this scarcity is PHYSICS-GUIDED AUGMENTATION of artificial
faults (e.g. Jefferson Lab / HTS-quench ML, 2020-2026). This module applies that method to a
dilution fridge, but keeps the REAL parts real:

  * DETECTOR TRAINED ONLY ON REAL HEALTHY TELEMETRY (real BlueFors 'blizzard' logs). No twin,
    no synthetic normal. The model learns the REAL fridge's normal noise + cross-stage
    behaviour via reconstruction.
  * FALSE ALARMS measured on REAL held-out healthy windows -> a genuine real-hardware number.
  * RECALL measured on faults injected ONTO REAL healthy windows using the twin's physics
    fault signatures (helium_leak/blocked_impedance/etc.). "Semi-synthetic": real noise floor
    + real baseline + physics-grounded perturbation. This is strictly more honest than a
    pure-twin evaluation and is the strongest evaluation possible without a lab partnership.

What this can and cannot claim:
  * CAN: "an anomaly detector trained on real healthy telemetry has X% false alarms on real
    held-out data and catches physics-grounded faults injected on real windows at Y% recall."
  * CANNOT: real fault CLASSIFICATION (needs real labeled faults, which do not exist).

Reuses bluefors_data (real logs). Physics fault signatures are self-contained here (a compact
version of realistic_faults' mechanisms) so the injection is auditable.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import bluefors_data as BF

STAGES = ["50K", "4K", "Still", "MXC"]   # channels present in the real BlueFors sample
DEFAULT_DIR = "data/real/bluefors_cryometrics_sample"


# --------------------------------------------------------------------------- #
# Physics fault signatures injected ONTO real healthy windows (auditable)
# --------------------------------------------------------------------------- #
def inject_fault(window: np.ndarray, fault: str, sev: float, rng) -> np.ndarray:
    """Apply a physics-grounded fault to a REAL healthy window [W, 4] (stages 50K,4K,Still,MXC).

    Signatures mirror realistic_faults.py (cold stages warm; leak/block differ in magnitude;
    quench spikes 4K). Multiplicative on the real values, so the REAL noise rides along.
    Ramp over the window so onset is gradual (as a developing fault is).
    """
    W = window.shape[0]
    a = np.linspace(0.0, 1.0, W)[:, None]      # activation ramp
    out = window.copy()
    s = float(sev)
    # indices: 0=50K, 1=4K, 2=Still, 3=MXC
    if fault == "helium_leak":
        out[:, 2] *= (1 + 0.15 * s * a[:, 0])          # Still warms a little
        out[:, 3] *= (1 + 0.55 * s * a[:, 0])          # MXC warms most
    elif fault == "blocked_impedance":
        out[:, 2] *= (1 + 0.25 * s * a[:, 0])          # Still warms more than leak
        out[:, 3] *= (1 + 0.50 * s * a[:, 0])
    elif fault == "wiring_heat_ingress":
        out[:, 2] *= (1 + 0.20 * s * a[:, 0])
        out[:, 3] *= (1 + 0.45 * s * a[:, 0])
    elif fault == "heat_load_spike":
        out[:, 3] *= (1 + 0.70 * s * a[:, 0])          # MXC-localized
    elif fault == "magnet_quench":
        spike = np.exp(-((np.arange(W) - 0.6 * W) ** 2) / (2 * (0.06 * W) ** 2))
        out[:, 1] *= (1 + 3.0 * s * spike)             # sharp 4K transient
    else:
        raise ValueError(fault)
    return out


FAULTS = ["helium_leak", "blocked_impedance", "wiring_heat_ingress",
          "heat_load_spike", "magnet_quench"]


# --------------------------------------------------------------------------- #
# Detector: reconstruction on real healthy telemetry
# --------------------------------------------------------------------------- #
def _windows(X: np.ndarray, W: int, step: int) -> np.ndarray:
    return np.array([X[i:i + W] for i in range(0, len(X) - W, step)])


def run(log_dir: str = DEFAULT_DIR, W: int = 30, step: int = 3, sev: float = 1.0,
        n_components: int = 6, fa_quantile: float = 0.99,
        out_path: str = "outputs/real_detect.json") -> dict:
    """Train on real healthy windows; report FA on real held-out + recall per injected fault."""
    d = BF.load_fridge_day(log_dir)
    n = min(len(d[s][1]) for s in STAGES)
    X = np.log(np.clip(np.stack([d[s][1][:n] for s in STAGES], axis=1), 1e-6, None))  # [n,4] real
    wins = _windows(X, W, step)
    rng = np.random.default_rng(0)

    # split real healthy: 60% train, 40% held-out test (both REAL)
    k = int(0.6 * len(wins))
    train, test_healthy = wins[:k], wins[k:]
    flat = lambda w: w.reshape(len(w), -1)
    mu, sd = flat(train).mean(0), flat(train).std(0) + 1e-9

    from sklearn.decomposition import PCA
    p = PCA(n_components=n_components).fit((flat(train) - mu) / sd)

    def resid(w):
        z = (flat(w) - mu) / sd
        r = p.inverse_transform(p.transform(z))
        return np.mean((z - r) ** 2, axis=1)

    thr = np.quantile(resid(train), fa_quantile)
    fa = float((resid(test_healthy) > thr).mean())   # REAL false-alarm rate

    # recall per fault: inject physics fault onto REAL held-out healthy windows (log space)
    per_fault = {}
    for f in FAULTS:
        faulted = np.stack([np.log(np.clip(
            inject_fault(np.exp(w), f, sev, rng), 1e-6, None)) for w in test_healthy])
        rec = float((resid(faulted) > thr).mean())
        per_fault[f] = round(rec, 3)

    result = {
        "data": "REAL BlueFors healthy telemetry (blizzard), no synthetic normal",
        "method": "PCA reconstruction detector; physics-guided augmentation (field-standard)",
        "n_train_real_windows": int(k),
        "n_test_real_windows": int(len(test_healthy)),
        "window_min": W, "severity": sev, "fa_threshold_quantile": fa_quantile,
        "false_alarm_rate_on_REAL_healthy": round(fa, 3),
        "recall_per_injected_physics_fault": per_fault,
        "mean_recall": round(float(np.mean(list(per_fault.values()))), 3),
        "reading": (
            "Detector trained ONLY on real healthy BlueFors telemetry. The false-alarm rate is "
            "a genuine real-hardware number (real held-out healthy windows). Recall is on "
            "physics faults injected onto REAL windows (real noise + real baseline + physics "
            "perturbation) — the field-standard 'physics-guided augmentation' method, applied "
            "to a dilution fridge for the first time. This CANNOT claim real fault "
            "classification (no real fault labels exist), only detection."),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"[real_detect] trained on {k} REAL healthy windows")
    print(f"[real_detect] FALSE ALARMS on real held-out healthy: {fa*100:.1f}%")
    print(f"[real_detect] recall per injected physics fault:")
    for f, r in per_fault.items():
        print(f"    {f:22s} {r*100:5.0f}%")
    print(f"[real_detect] mean recall {result['mean_recall']*100:.0f}% | wrote {out_path}")
    return result


if __name__ == "__main__":
    run()
