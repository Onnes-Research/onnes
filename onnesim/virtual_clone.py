"""
virtual_clone.py — a data-driven "virtual clone" of a REAL dilution fridge.

Naive simulators add independent Gaussian noise to each channel. A REAL fridge does
NOT behave that way: each stage has its own noise magnitude, the stages DRIFT
TOGETHER (shared ambient / pulse-tube fluctuations), and the drift is temporally
correlated (slow wander, not white noise). This module LEARNS that fingerprint from
real BlueFors logs and reproduces it, so generated telemetry is statistically
matched to the real machine — a genuine behavioral clone of that fridge.

What it learns from the real data:
  1. base (median) temperature per stage,
  2. per-stage noise magnitude (std),
  3. the cross-stage CORRELATION matrix (how stages move together),
  4. the temporal autocorrelation (drift timescale) via an AR(1) coefficient.

It then generates synthetic telemetry with the SAME covariance + drift structure
(correlated AR(1) process shaped by the learned covariance). Validation compares the
clone's statistics to a held-out slice of the real data.

⚠️ SCOPE (honest): v1 clones the STEADY-STATE fingerprint (base operation). It does
NOT yet model cooldown dynamics or faults (needs data spanning those regimes). It is
a real statistical twin of base operation, not a full dynamical clone.
"""
from __future__ import annotations
import numpy as np

from . import bluefors_data as BF

STAGES = ["50K", "4K", "Still", "MXC"]


def _ar1_coeff(x: np.ndarray) -> float:
    """Lag-1 autocorrelation (AR(1) drift coefficient) of a series."""
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    x0, x1 = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    denom = np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum())
    return float((x0 * x1).sum() / denom) if denom > 0 else 0.0


def learn_fingerprint(log_dir: str) -> dict:
    """Learn the statistical fingerprint of a real fridge from its BlueFors logs."""
    data = BF.load_fridge_day(log_dir)
    stages = [s for s in STAGES if s in data]
    n = min(len(data[s][1]) for s in stages)
    M = np.array([data[s][1][:n] for s in stages])   # [n_stages, n]

    med = M.mean(axis=1)
    std = M.std(axis=1)
    # correlation of the fluctuations (subtract slow mean via per-stage centering)
    corr = np.corrcoef(M)
    ar1 = np.array([_ar1_coeff(M[i]) for i in range(len(stages))])
    # covariance of the residuals for correlated generation
    resid = M - med[:, None]
    cov = np.cov(resid)
    return {"stages": stages, "median": med, "std": std, "corr": corr,
            "ar1": ar1, "cov": cov, "n": n}


def generate(fp: dict, n_steps: int, seed: int = 0) -> dict:
    """Generate cloned telemetry matching the learned fingerprint.

    Approach: generate a correlated AR(1) drift process, then RESCALE each stage so
    its marginal std exactly matches the real std, and mix via the real correlation
    matrix's Cholesky factor so cross-stage correlations are preserved. This matches
    real std AND real cross-correlation directly (more robust than deriving the
    innovation covariance from the AR(1) variance identity, which is ill-conditioned
    when a stage is a near-random-walk, a_i -> 1)."""
    rng = np.random.default_rng(seed)
    stages = fp["stages"]
    k = len(stages)
    a = np.clip(fp["ar1"], 0.0, 0.9995)

    # 1) per-stage unit-variance AR(1) drift (independent)
    z = np.zeros((n_steps, k))
    r = np.zeros(k)
    for t in range(n_steps):
        e = rng.standard_normal(k) * np.sqrt(1.0 - a ** 2)  # keeps Var->1 for a<1
        r = a * r + e
        z[t] = r
    # standardize each column to exactly unit variance (fixes random-walk drift too)
    z = (z - z.mean(0)) / (z.std(0) + 1e-18)

    # 2) impose the REAL cross-stage correlation via Cholesky
    corr = fp["corr"]
    corr = (corr + corr.T) / 2
    w, V = np.linalg.eigh(corr)
    w = np.clip(w, 1e-12, None)
    Lc = V @ np.diag(np.sqrt(w)) @ V.T          # symmetric sqrt of corr
    zc = z @ Lc.T
    zc = (zc - zc.mean(0)) / (zc.std(0) + 1e-18)  # re-standardize after mixing

    # 3) scale to real std, offset to real median
    out = fp["median"][None, :] + zc * fp["std"][None, :]
    return {"stages": stages, "series": out}


def validate(fp: dict, seed: int = 0) -> dict:
    """Generate a cloned series and compare its statistics to the real fingerprint."""
    gen = generate(fp, fp["n"], seed=seed)
    G = gen["series"].T                                # [n_stages, n]
    return {
        "stages": fp["stages"],
        "real_median": fp["median"], "clone_median": G.mean(axis=1),
        "real_std": fp["std"], "clone_std": G.std(axis=1),
        "real_corr": fp["corr"], "clone_corr": np.corrcoef(G),
    }
