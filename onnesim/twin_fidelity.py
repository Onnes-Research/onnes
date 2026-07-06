"""
twin_fidelity.py — a NUMBER for how "real-like" the twin is (discriminator AUC).

WHY THIS EXISTS
---------------
The paper says the twin is "fingerprinted to real BlueFors logs" but never quantifies
fidelity. Reviewers (and Section 9) ask the sim-to-real question directly. The cheapest
decisive answer is a two-sample discriminator test (a GAN-style critic, but measured not
trained adversarially):

    Train a classifier to tell a REAL BlueFors telemetry window from a TWIN window.
      * AUC ~ 0.5  -> the twin is statistically INDISTINGUISHABLE from the real fridge
                     on these features (the strong fidelity claim, now with a number).
      * AUC ~ 1.0  -> trivially distinguishable; the feature IMPORTANCES then say exactly
                     which channel/statistic to fix.

This is the standard "classifier two-sample test" (Lopez-Paz & Oquab, 2017) applied to a
cryogenic digital twin. It needs NO fault labels and NO API calls — it runs fully offline
from the real logs already in data/real/.

HONEST SCOPE
------------
The real telemetry is a single ~24 h BlueFors run (one fridge, one day), so real windows
are autocorrelated; we cut them with a stride and report the caveat. The twin windows are
independent draws. We report TWO conditions:
  * "absolute"     — includes per-channel levels (mean/start/end). A twin whose BASE
                     TEMPERATURES are slightly off is separable here; that is a real,
                     fixable defect, so we surface it.
  * "fingerprint"  — level-invariant features only (relative std, lag-1 autocorrelation,
                     normalized slope). This isolates the NOISE/CORRELATION fingerprint
                     claim from any base-temperature offset.
The honest headline is the fingerprint-condition AUC: it measures the thing the paper
actually claims to reproduce.

Reuses (nothing re-implemented): bluefors_data.load_fridge_day (the real loader),
virtual_clone.learn_fingerprint (the same fingerprint the engine uses), and
cryo_engine.simulate (the twin). Writes outputs/twin_fidelity.json.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from . import bluefors_data as BF
from . import cryo_engine as CE
from . import virtual_clone as VC
from . import stats as ST

# Real BlueFors channels map to the twin's FRTMS schema names. The real sample has no
# cold-plate (CP) channel, so the discriminator uses the four shared stages + flow.
REAL_TO_TWIN = {"50K": "temp1_T", "4K": "temp2_T", "Still": "temp3_T", "MXC": "temp5_T"}
DISC_CHANNELS = ["temp1_T", "temp2_T", "temp3_T", "temp5_T", "flowmeter"]
DEFAULT_FINGERPRINT_DIR = "data/real/bluefors_cryometrics_sample"


# --------------------------------------------------------------------------- #
# Feature extraction — two feature sets: absolute (includes level) and fingerprint
# (level-invariant, isolates the noise/correlation claim).
# --------------------------------------------------------------------------- #
def _lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    x0, x1 = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    d = np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum())
    return float((x0 * x1).sum() / d) if d > 0 else 0.0


def _window_features(win: dict, mode: str) -> tuple[list[float], list[str]]:
    """Per-channel features for one window. mode in {'absolute','fingerprint'}.

    absolute    -> mean, std, normalized slope, relative-std, lag1 autocorr
    fingerprint -> relative-std, lag1 autocorr, normalized slope  (NO absolute level)
    """
    feats: list[float] = []
    names: list[str] = []
    n = len(win["t_s"])
    tnorm = np.linspace(0.0, 1.0, n)
    for ch in DISC_CHANNELS:
        x = np.asarray(win[ch], dtype=float)
        m = np.isfinite(x)
        if m.sum() < 3:
            x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.any(m) else 0.0)
            m = np.ones(n, dtype=bool)
        xf = x[m]
        med = float(np.median(xf))
        mean = float(np.mean(xf))
        std = float(np.std(xf))
        rel_std = std / (abs(mean) + 1e-12)
        slope = float(np.polyfit(tnorm[m], xf, 1)[0]) if np.ptp(tnorm[m]) > 0 else 0.0
        nslope = slope / (abs(mean) + 1e-12)
        ac = _lag1_autocorr(xf)
        if mode == "absolute":
            feats += [mean, std, nslope, rel_std, ac]
            names += [f"{ch}_mean", f"{ch}_std", f"{ch}_nslope", f"{ch}_relstd", f"{ch}_ac1"]
        else:  # fingerprint: level-invariant only
            feats += [rel_std, ac, nslope]
            names += [f"{ch}_relstd", f"{ch}_ac1", f"{ch}_nslope"]
    return feats, names


# --------------------------------------------------------------------------- #
# Window sources
# --------------------------------------------------------------------------- #
def real_windows(log_dir: str, win: int, stride: int) -> list[dict]:
    """Cut overlapping windows from the real 24 h BlueFors run, mapped to twin schema.

    Only the shared channels are populated (four stages + flow); the real sample has no
    cold-plate channel. Windows carry a synthetic t_s at 1-min cadence (the BlueFors rate).
    """
    d = BF.load_fridge_day(log_dir)
    stages = [s for s in REAL_TO_TWIN if s in d]
    if "flow" not in d:
        raise RuntimeError("real sample missing flow channel")
    n = min(len(d[s][1]) for s in stages + ["flow"])
    series = {REAL_TO_TWIN[s]: np.asarray(d[s][1][:n], dtype=float) for s in stages}
    series["flowmeter"] = np.asarray(d["flow"][1][:n], dtype=float)
    out = []
    for i0 in range(0, n - win + 1, stride):
        w = {"t_s": np.arange(win) * 60.0}
        for ch in DISC_CHANNELS:
            w[ch] = series[ch][i0:i0 + win].copy() if ch in series else np.zeros(win)
        out.append(w)
    return out


def twin_windows(n_windows: int, win: int, fingerprint: dict | None,
                 base_seed: int = 40_000) -> list[dict]:
    """Generate independent twin NORMAL windows at 1-min cadence matching the real length.

    Seeds are disjoint from every seed range used elsewhere (train 0.., demos 500..,
    eval 10_000.., ablation 20_000..) so the fidelity test never reuses a paper scenario.
    """
    cfg = CE.EngineConfig(fingerprint=fingerprint, realistic=True, imperfections=True)
    hours = win / 60.0
    out = []
    for j in range(n_windows):
        cols = CE.simulate(CE.Scenario("normal", 0.0, 0.4), cfg,
                           hours=hours, dt_min=1.0, seed=base_seed + j)
        # keep exactly `win` samples on the shared channels
        w = {"t_s": np.asarray(cols["t_s"][:win], dtype=float)}
        for ch in DISC_CHANNELS:
            w[ch] = np.asarray(cols[ch][:win], dtype=float)
        out.append(w)
    return out


# --------------------------------------------------------------------------- #
# The classifier two-sample test
# --------------------------------------------------------------------------- #
def _auc_cv(X: np.ndarray, y: np.ndarray, seed: int = 0, repeats: int = 5, folds: int = 5):
    """Repeated stratified k-fold ROC-AUC for a RandomForest discriminator.

    Returns (mean_auc, std_auc, per_fold list, fitted-on-all importances). Small-n aware:
    AUC is averaged over repeats*folds held-out folds; we also report a permutation null.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    aucs = []
    for tr, te in rskf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        # a fold can be single-class in tiny sets; guard
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[te], p)))
    clf_all = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    return (float(np.mean(aucs)) if aucs else 0.5,
            float(np.std(aucs)) if aucs else 0.0,
            aucs, clf_all.feature_importances_)


def _permutation_null(X: np.ndarray, y: np.ndarray, seed: int, n_perm: int = 50) -> float:
    """Mean AUC under label permutation — the empirical null (~0.5 if the test is calibrated)."""
    rng = np.random.default_rng(seed + 1)
    vals = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        m, _s, _f, _i = _auc_cv(X, yp, seed=seed, repeats=1, folds=5)
        vals.append(m)
    return float(np.mean(vals))


def run(log_dir: str = DEFAULT_FINGERPRINT_DIR, win: int = 180, stride: int = 30,
        seed: int = 0, out_path: str = "outputs/twin_fidelity.json") -> dict:
    """Compute discriminator AUC (real vs twin) in both feature conditions + write JSON.

    win/stride are in 1-min samples (win=180 -> 3 h windows). The twin is fingerprinted to
    THIS fridge, so a low AUC is a strong statement: even a critic cannot separate them.
    """
    fp = None
    try:
        fp = VC.learn_fingerprint(log_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[fidelity] fingerprint unavailable ({exc}); twin uses engine noise")

    real = real_windows(log_dir, win=win, stride=stride)
    twin = twin_windows(n_windows=len(real), win=win, fingerprint=fp)
    print(f"[fidelity] {len(real)} real windows vs {len(twin)} twin windows "
          f"(win={win} min, stride={stride} min)")

    result = {
        "config": {"log_dir": log_dir, "win_min": win, "stride_min": stride,
                   "n_real": len(real), "n_twin": len(twin),
                   "channels": DISC_CHANNELS, "fingerprint": fp is not None},
        "conditions": {},
        "reading": (
            "Classifier two-sample test (Lopez-Paz & Oquab 2017) on real BlueFors vs twin "
            "windows. AUC~0.5 => twin indistinguishable from the real fridge on these "
            "features; AUC~1.0 => separable, and the top feature importances say what to "
            "fix. 'fingerprint' (level-invariant) is the honest headline; 'absolute' also "
            "sees base-temperature offsets. Real windows come from one 24 h run (correlated); "
            "n is small, so AUC carries a permutation null and a spread."),
    }

    for mode in ("absolute", "fingerprint"):
        Xr = [_window_features(w, mode)[0] for w in real]
        Xt = [_window_features(w, mode)[0] for w in twin]
        names = _window_features(real[0], mode)[1]
        X = np.asarray(Xr + Xt, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray([1] * len(Xr) + [0] * len(Xt))  # 1 = real, 0 = twin
        mean_auc, std_auc, folds, imp = _auc_cv(X, y, seed=seed)
        null_auc = _permutation_null(X, y, seed=seed)
        order = np.argsort(imp)[::-1][:6]
        top = [{"feature": names[i], "importance": round(float(imp[i]), 4)} for i in order]
        # a crude separability CI: treat mean over folds with its spread
        result["conditions"][mode] = {
            "auc_mean": round(mean_auc, 3),
            "auc_std": round(std_auc, 3),
            "auc_permutation_null": round(null_auc, 3),
            "excess_over_null": round(mean_auc - null_auc, 3),
            "n_folds_scored": len(folds),
            "top_discriminative_features": top,
            "interpretation": _interpret(mean_auc),
        }
        print(f"[fidelity] {mode:11s} AUC={mean_auc:.3f}±{std_auc:.3f} "
              f"(null {null_auc:.3f}) top: {top[0]['feature']}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fidelity] wrote {out_path}")
    return result


def _interpret(auc: float) -> str:
    if auc < 0.65:
        return "indistinguishable-to-weak: strong fidelity (critic barely beats chance)"
    if auc < 0.85:
        return "partially separable: fingerprint is close but a feature leaks; see top features"
    return "separable: the twin is distinguishable; top features name the defect to fix"


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
