"""
real_data.py — load REAL labeled multivariate telemetry (not synthetic).

Purpose: prove the OnnesSim ML/agent pipeline works on REAL sensor-anomaly data,
as a transfer check. The canonical fridge fault corpus does not exist publicly
(see docs/REAL_DATA_OPTIONS.md), so we validate the METHOD on real labeled
telemetry from adjacent domains and say so plainly.

⚠️ HONESTY: these are NOT dilution-fridge data. Server/industrial telemetry has
the same SHAPE as fridge telemetry (multivariate time series + per-timestep
anomaly labels), so it validates the detection MACHINERY, not any cryogenic claim.
Never report results here as fridge performance.

Verified downloadable sources (links resolve as of 2026-07):
- Server Machine Dataset (SMD), NetManAIOps/OmniAnomaly (MIT):
  https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
  38 channels, per-timestep 0/1 labels. Download train/<m>.txt + test_label/<m>.txt.
- NASA SMAP/MSL telemetry anomalies: https://github.com/khundman/telemanom
"""
from __future__ import annotations
import os
import numpy as np


def load_smd(train_path: str, label_path: str | None = None,
             window: int = 100, stride: int = 50) -> dict:
    """Load a Server Machine Dataset file into windows with a per-window label.

    A window is labeled anomalous if ANY timestep in it is labeled 1 (standard
    point-adjust-free convention for window classification).

    Returns {X_windows: [n,window*channels], y: [n], n_channels, raw_labels}.
    """
    X = np.loadtxt(train_path, delimiter=",")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_ch = X.shape[1]

    y_ts = None
    if label_path and os.path.exists(label_path):
        y_ts = np.loadtxt(label_path)

    feats, labels = [], []
    for start in range(0, len(X) - window + 1, stride):
        w = X[start:start + window]
        # simple, model-agnostic features per channel: mean, std, min, max, slope
        f = []
        t = np.linspace(0, 1, window)
        for c in range(n_ch):
            col = w[:, c]
            slope = float(np.polyfit(t, col, 1)[0]) if np.ptp(col) > 0 else 0.0
            f.extend([col.mean(), col.std(), col.min(), col.max(), slope])
        feats.append(f)
        if y_ts is not None:
            labels.append(int(y_ts[start:start + window].max() > 0))

    return {
        "X_windows": np.asarray(feats, dtype=float),
        "y": np.asarray(labels, dtype=int) if labels else None,
        "n_channels": n_ch,
        "n_windows": len(feats),
    }
