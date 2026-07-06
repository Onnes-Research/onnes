"""
cryowala_scenarios.py — physics-grounded fault scenarios via the REAL Cryowala map.

Unlike faults.py (phenomenological multipliers on a placeholder thermal model), here
a fault is a PHYSICAL change in per-stage HEAT LOAD [W], and the resulting stage
temperatures come from the BSD-3 CryowalaCore-calibrated T_funcs (cryowala_physics).
This makes each scenario defensible: "we added X uW to stage Y, the calibrated map
says the fridge warms to Z" — traceable to a published, real-fridge-fit model.

Fault -> heat-load mechanism (all in Watts, perturbing NOMINAL_LOADS_W):
  normal              — loads at nominal
  heat_load_spike     — extra load on a cold stage (CP/MXC)
  helium_leak         — cooling degraded == effective load up on Still+CP+MXC
  magnet_quench       — large transient load on the 4K flange (magnet thermalized there)
  wiring_heat_ingress — extra passive load spread down the chain (control-wiring at scale)
  blocked_impedance   — circulation restricted == load up on Still/MXC

Emits the same FRTMS-style schema as telemetry.py so it's a drop-in for the agent/ML.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import cryowala_physics as CP

SCENARIO_CLASSES = [
    "normal", "heat_load_spike", "helium_leak",
    "magnet_quench", "wiring_heat_ingress", "blocked_impedance",
]
# stage indices
_50K, _4K, _STILL, _CP, _MXC = 0, 1, 2, 3, 4


@dataclass
class ScenarioSpec:
    fault_class: str
    severity: float          # 0..1
    onset_frac: float = 0.5  # when in the window the fault ramps in


def _fault_loads(fault_class: str, sev: float, active: float) -> np.ndarray:
    """Return per-stage heat loads [W] for a fault at activation level `active` in [0,1]."""
    loads = CP.NOMINAL_LOADS_W.copy()
    s = sev * active
    if fault_class == "normal":
        pass
    elif fault_class == "heat_load_spike":
        loads[_MXC] += 40e-6 * s          # up to +40 uW on MXC
    elif fault_class == "helium_leak":
        loads[[_STILL, _CP, _MXC]] *= (1 + 3.0 * s)   # cooling loss ~ load rise
    elif fault_class == "magnet_quench":
        loads[_4K] += 0.4 * s             # large transient onto 4K flange
    elif fault_class == "wiring_heat_ingress":
        loads[[_STILL, _CP, _MXC]] += np.array([5e-3, 30e-6, 8e-6]) * s
    elif fault_class == "blocked_impedance":
        loads[[_STILL, _MXC]] *= (1 + 4.0 * s)
    return loads


def simulate_scenario(spec: ScenarioSpec, hours: float = 6.0, dt_min: float = 5.0,
                      seed: int = 0) -> dict:
    """Generate a telemetry window for a scenario using the REAL Cryowala T_funcs.

    The fridge is assumed settled-cold; the fault ramps in after onset. Each timestep
    computes stage temps from the (possibly perturbed) heat loads via cryowala_physics.
    Returns FRTMS-schema columns (temp1_T..temp8_T, p*, flow, statuses).
    """
    rng = np.random.default_rng(seed)
    n = int(hours * 60 / dt_min)
    t_s = np.arange(n) * dt_min * 60.0
    onset = spec.onset_frac * t_s[-1] if n > 1 else 0.0

    temps = np.zeros((n, 5))
    for k in range(n):
        active = 0.0 if t_s[k] <= onset else min(1.0, (t_s[k] - onset) / (0.3 * t_s[-1] + 1))
        loads = _fault_loads(spec.fault_class, spec.severity, active)
        temps[k] = CP.all_stage_temps(loads)

    def noisy(x, rel=0.01):
        return x * (1 + rel * rng.standard_normal(np.shape(x)))

    cols = {"t_s": t_s}
    for i in range(5):
        cols[f"temp{i+1}_T"] = noisy(temps[:, i])
    # magnet sensors track the 4K flange (temp2); quench shows here first
    cols["temp6_T"] = noisy(temps[:, _4K] + 0.05)
    cols["temp7_T"] = noisy(temps[:, _4K] + 0.05)
    cols["temp8_T"] = np.full(n, 300.0)  # spare channel
    # flow proxy: drops for blocked_impedance
    flow = np.full(n, 0.8)
    if spec.fault_class == "blocked_impedance":
        flow = np.where(t_s > onset, 0.8 * (1 - 0.5 * spec.severity), 0.8)
    cols["flowmeter"] = np.maximum(noisy(flow, 0.02), 0.0)
    for j in range(6):
        cols[f"p{j+1}"] = np.maximum(noisy(np.full(n, [1.5e3, 0.35, 8, 1e3, 2e-2, 5e2][j], ), 0.015), 0)
    cols["cpa_status"] = np.ones(n, dtype=int)
    cols["turbo_status"] = np.ones(n, dtype=int)
    return cols


def sample_spec(rng: np.random.Generator) -> ScenarioSpec:
    fc = rng.choice(SCENARIO_CLASSES)
    sev = 0.0 if fc == "normal" else float(rng.uniform(0.3, 1.0))
    return ScenarioSpec(fault_class=fc, severity=sev, onset_frac=float(rng.uniform(0.3, 0.6)))
