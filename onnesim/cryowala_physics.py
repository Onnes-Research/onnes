"""
cryowala_physics.py — real dilution-fridge heat-load -> temperature physics.

Reimplemented from CryowalaCore (github.com/CirQuS-UTS/CryowalaCore), BSD-3-Clause,
Copyright (c) 2024 CirQuS UTS. This is a faithful port of that project's empirically
CALIBRATED model: `T_funcs(i, x)` maps the vector of per-stage heat loads x (Watts)
to the temperature of stage i (Kelvin), using coefficients fitted to a REAL fridge
("fridge_ours") in the Cryowala repo.

Why this matters for Onnes: it replaces OnnesSim's phenomenological PLACEHOLDER
cooling with a real, published, fridge-calibrated heat-load->temperature relation.
Faults become physically meaningful (a fault = a real change in heat load), and a
neural surrogate can be trained to emulate this map at fleet scale (see ml_surrogate.py).

Attribution: coefficients `coeffs`, `p0`, `t0` and the `T_funcs`/`q` forms are from
CryowalaCore/param_functions.py (BSD-3-Clause). Stage order: [50K, 4K, Still, CP, MXC].
"""
from __future__ import annotations
import numpy as np

STAGE_LABELS = ["50K", "4K", "Still", "CP", "MXC"]

# --- fridge_ours: real fridge parameters (CryowalaCore, BSD-3) --------------
FRIDGE_OURS = {
    "labels": STAGE_LABELS,
    "temps": np.array([46, 3.94, 1.227, 0.150, 0.020]),      # no-load steady temps [K]
    "lengths": np.array([0.236, 0.337, 0.3, 0.15, 0.222]),    # inter-stage lengths [m]
    "cool_p": np.array([10, 0.5, 30e-3, 300e-6, 20e-6]),      # cooling power [W]
}

# --- T_funcs fitted coefficients (CryowalaCore/param_functions.py, BSD-3) ---
_COEFFS = np.array([
    [[-2.84462757e-2, 6.14328293e-1, 0, 0], [-4.91928663e-3, 1.89496142e-3, 0, 0],
     [-2.29812049e-2, 3.25335785e-2, 0, 0], [2.04991379e-2, -1.78944956e-4, 0, 0],
     [1.66500853e-2, 1.68560160e-4, 0, 0]],
    [[2.03577334e-3, 2.33226365e-2, 0, 0], [1.52400937e-3, 2.33839654e-3, 0, 0],
     [2.16019081e-3, 6.47209581e-3, 0, 0], [-2.15628081e-4, 2.35905941e-5, 0, 0],
     [1.05157802e-3, -3.40602022e-4, 0, 0]],
    [[7.92949604e-5, 1.41004249e-3, 0, 0], [-1.02478043e-3, 8.63523085e-5, 0, 0],
     [0, 0, 7.83759658e-2, 1.79787617e1], [-2.14938424e-4, -1.73050413e-6, 0, 0],
     [1.97263046e-4, -1.13826468e-4, 0, 0]],
    [[5.42210196e-6, 3.97531363e-5, 0, 0], [-8.59342240e-6, 2.68716951e-6, 0, 0],
     [5.73569662e-7, 5.17351534e-5, 0, 0], [1.89729062e-5, 7.15643349e-5, 0, 0],
     [1.99493827e-6, 2.08526457e-5, 0, 0]],
    [[-3.49658618e-7, 8.56669958e-6, 0, 0], [-6.95765413e-6, 4.93801030e-7, 0, 0],
     [0, 0, -8.83265839e-4, 1.61380894e1], [-3.36859606e-6, 3.10642203e-7, 0, 0],
     [0, 0, 4.43619976e-3, 1.39739117e1]],
])
_P0 = np.array([1, 100, 30, 60, 12])
_T0 = np.array([4.20111000e1, 3.41026571e0, 1.20520821e0, 1.63359036e-1, 1.64774143e-2])
# Heat-load unit scaling to match the fit (W, mW, mW, uW, uW conventions).
_SCALE = np.array([1, 1e3, 1e3, 1e6, 1e6])


def _q(x, a, b):
    return np.nan_to_num(a * (np.sqrt(x + b) - np.sqrt(b)), nan=-a * np.sqrt(b))


def stage_temp(i: int, loads_w: np.ndarray) -> float:
    """Temperature [K] of stage i given per-stage heat loads (Watts).
    Faithful port of CryowalaCore T_funcs(i, x)."""
    x = np.asarray(loads_w, dtype=float) * _SCALE - _P0
    y = np.sum(x * _COEFFS[i, :, 1] + _q(x, _COEFFS[i, :, 2], _COEFFS[i, :, 3]))
    return float(_T0[i] + y)


def all_stage_temps(loads_w: np.ndarray) -> np.ndarray:
    """Temperatures [K] for all 5 stages given a heat-load vector (Watts)."""
    return np.array([stage_temp(i, loads_w) for i in range(5)])


def is_physical(temps: np.ndarray) -> bool:
    """The Cryowala polynomial fit is only calibrated near its operating point and
    can return unphysical (e.g. negative, or non-monotonic) temperatures when
    extrapolated to extreme loads. A result is physical only if every stage is
    positive and stages are ordered warm->cold (50K > 4K > Still > CP > MXC)."""
    t = np.asarray(temps)
    return bool(np.all(t > 0) and np.all(np.diff(t) < 0))


# Nominal (baseline) per-stage heat loads that reproduce ~fridge_ours temps.
# Derived to sit near the fit's set-point p0 (in native W): p0 / _SCALE.
NOMINAL_LOADS_W = _P0 / _SCALE  # [1 W, 0.1 W, 30 mW, 60 uW, 12 uW]
