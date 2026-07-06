"""
timesfm_opponent.py — Google TimesFM as a foundation-model DETECTION opponent.

WHAT THIS IS (and honestly is NOT)
----------------------------------
TimesFM (github.com/google-research/timesfm, 2.5 200M) is a zero-shot time-series
FORECASTER, not a classifier. So it cannot be a drop-in 6-class fault classifier like the
random forest. What it CAN do — and the honest way to use a forecaster as a fault opponent —
is forecast-residual anomaly DETECTION:

  1. Split each telemetry window into a context (first part) and a holdout (tail).
  2. TimesFM forecasts the tail of each channel from the context, zero-shot (no training).
  3. If the real tail deviates from the forecast beyond a calibrated threshold on any
     safety channel, flag an anomaly (fault detected).

This yields a NO-TRAINING foundation-model DETECTION baseline directly comparable to the
agent's and RF's detection F1, under the SAME seed-addressed scenarios. It is NOT a
classification opponent — no forecaster is — and we say so: it is evaluated on detection
(normal vs any-fault) only. The threshold is calibrated on NORMAL scenarios from the train
seed range (0..), disjoint from the eval seeds (10_000..), so there is no leakage.

COMPUTE
-------
TimesFM 2.5 is 200M params; it downloads weights from HuggingFace on first use and runs on
CPU (slow) or GPU (ONNES_DEVICE=cuda / the H200). CPU-verifiable at tiny n with --smoke.

Reuses cryo_engine (twin), benchmark.sample_specs (identical seeds), evaluate.score (same
detection metric as the agent). Writes outputs/timesfm_opponent.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import cryo_engine as CE
from . import benchmark as BM
from . import evaluate as E

NORMAL = "normal"
# Safety channels the detector watches (cold stages + flow + the OVC/condenser pressures
# that separate the confusable faults). Forecast residual on ANY of these can trip detection.
WATCH_CHANNELS = ["temp3_T", "temp4_T", "temp5_T", "flowmeter", "p1", "p5"]


def _load_timesfm():
    """Load + compile TimesFM 2.5 (200M, torch). Cached module-global. GPU if ONNES_DEVICE=cuda.

    Uses the confirmed 2.5 API: from_pretrained -> compile(ForecastConfig) -> forecast().
    """
    global _TFM
    try:
        return _TFM
    except NameError:
        pass
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=512, max_horizon=64,
        normalize_inputs=True, use_continuous_quantile_head=True,
    ))
    _TFM = model
    return _TFM


def _window_residual_scores(cols: dict, model, context_frac: float = 0.6,
                            horizon: int = 16) -> dict:
    """For each watch channel, forecast the tail from the context and return the max
    standardized residual (|real - forecast| / context_std) over the horizon.

    A high score on any channel means the channel's actual continuation departed from what a
    zero-shot forecaster expected of a normal series — the anomaly signal.
    """
    inputs, keys = [], []
    horizons = []
    for ch in WATCH_CHANNELS:
        if ch not in cols:
            continue
        x = np.asarray(cols[ch], dtype=float)
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0)
        n = len(x)
        cut = max(8, int(n * context_frac))
        h = min(horizon, n - cut)
        if h < 2:
            continue
        inputs.append(x[:cut])
        keys.append((ch, cut, h, x))
        horizons.append(h)
    if not inputs:
        return {}
    H = max(horizons)
    point_fc, _ = model.forecast(horizon=H, inputs=[a.astype(np.float32) for a in inputs])
    scores = {}
    for i, (ch, cut, h, x) in enumerate(keys):
        fc = np.asarray(point_fc[i])[:h]
        real = x[cut:cut + h]
        ctx_std = np.std(x[:cut]) + 1e-9
        resid = np.abs(real - fc) / ctx_std
        scores[ch] = float(np.max(resid))
    return scores


def _calibrate_threshold(model, n_normal: int, base_seed: int, sev_scale: float,
                         quantile: float = 0.95) -> float:
    """Pick the detection threshold as a high quantile of the max-residual on NORMAL windows
    from the TRAIN seed range (disjoint from eval). This sets the false-alarm operating point
    without ever seeing eval data — the honest calibration."""
    cfg = CE.EngineConfig(realistic=True, imperfections=True)
    specs = BM.sample_specs(n_normal * 4, base_seed=base_seed)
    normal_specs = [s for s in specs if s.fault_class == NORMAL][:n_normal]
    if not normal_specs:  # ensure some normals
        normal_specs = [BM.Spec(base_seed + i, NORMAL, 0.0, 0.4) for i in range(n_normal)]
    maxres = []
    for s in normal_specs:
        cols = CE.simulate(CE.Scenario(NORMAL, 0.0, 0.4), cfg, hours=6.0, dt_min=5.0, seed=s.seed)
        sc = _window_residual_scores(cols, model)
        if sc:
            maxres.append(max(sc.values()))
    return float(np.quantile(maxres, quantile)) if maxres else 3.0


def run(n_eval: int = 60, n_calib: int = 20, base_seed_eval: int = 10_000,
        base_seed_calib: int = 0, sev_scale: float = 0.5, smoke: bool = False,
        out_path: str = "outputs/timesfm_opponent.json") -> dict:
    """Score TimesFM forecast-residual DETECTION on the eval seeds; compare to agent/RF F1."""
    if smoke:
        n_eval, n_calib = 8, 6
    model = _load_timesfm()
    print(f"[timesfm] calibrating threshold on {n_calib} NORMAL train-seed windows ...")
    thr = _calibrate_threshold(model, n_calib, base_seed_calib, sev_scale)
    print(f"[timesfm] detection threshold (95th pct normal max-residual) = {thr:.2f}")

    cfg = CE.EngineConfig(realistic=True, imperfections=True)
    specs = BM.sample_specs(n_eval, base_seed=base_seed_eval)
    rows = []
    for i, s in enumerate(specs):
        cols = CE.simulate(CE.Scenario(s.fault_class, s.severity * sev_scale, s.onset_frac),
                           cfg, hours=6.0, dt_min=5.0, seed=s.seed)
        sc = _window_residual_scores(cols, model)
        detected = bool(sc and max(sc.values()) > thr)
        rows.append({"truth_class": s.fault_class,
                     "pred_detected": detected,
                     "pred_class": NORMAL})  # forecaster does DETECTION only, no class
        if smoke:
            print(f"  [{i+1}/{n_eval}] truth={s.fault_class:18s} maxres="
                  f"{(max(sc.values()) if sc else 0):5.2f} -> detected={detected}")
    score = E.score(rows)
    result = {
        "model": "google/timesfm-2.5-200m-pytorch",
        "task": "detection_only (forecast-residual; TimesFM is a forecaster, not a classifier)",
        "n_eval": n_eval, "threshold": round(thr, 3),
        "detection": {"f1": score["f1"], "precision": score["precision"],
                      "recall": score["recall"], "accuracy": score["detect_accuracy"]},
        "counts": score["counts"],
        "reading": (
            "TimesFM zero-shot forecast-residual anomaly detection under the SAME eval seeds "
            "as the agent head-to-head. It flags a fault when any watch channel's real "
            "continuation departs from TimesFM's forecast beyond a threshold calibrated on "
            "NORMAL train-seed windows. This is a NO-TRAINING foundation-model DETECTION "
            "opponent — directly comparable on detection F1, but it does not classify the "
            "fault (no forecaster does). Compare to agent detection F1 0.979 / RF 0.997."),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[timesfm] detection F1={score['f1']}  P={score['precision']}  R={score['recall']}")
    print(f"[timesfm] wrote {out_path}")
    return result


if __name__ == "__main__":
    import sys
    run(smoke="--smoke" in sys.argv)
