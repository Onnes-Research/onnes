"""
cryo_engine.py — the unified "virtual cryo-cooling engine".

Combines the three real ingredients this project built:
  1. REAL physics: the T^2 dilution-cooling floor (dilution_cooling.py) -> base temps
     that respond correctly to heat load, validated to the LD400 calibration.
  2. REAL fridge behavior: the virtual clone's learned noise + cross-stage
     correlations (virtual_clone.py), fit to real BlueFors logs -> telemetry that
     looks like a real machine, not toy Gaussian noise.
  3. Fault injection as real heat-load perturbations -> labeled scenarios whose
     temperatures come from the physics, carrying a real fridge's noise character.

This is the engine agents and ML models are tested against. It is the honest
"virtual clone of a cryo cooler for quantum computers": real physics for the mean,
a real fridge's fingerprint for the fluctuations, physics-grounded faults for labels.

Emits the verified FRTMS/Bluefors schema (temp1..temp5 stages, flow) so it is a
drop-in for the agent + ML pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .dilution_cooling import DilutionUnit
from . import virtual_clone as VC

STAGES = ["50K", "4K", "Still", "CP", "MXC"]

# Nominal parasitic heat loads per stage [W] that reproduce a real base state.
# MXC/CP tuned to the T^2 floor: ~5 uW -> ~11 mK (BlueFors), CP ~ real 0.1 K.
NOMINAL_LOADS_W = np.array([15.0, 0.30, 8.0e-4, 30e-6, 5.0e-6])

FAULT_CLASSES = ["normal", "heat_load_spike", "helium_leak",
                 "magnet_quench", "wiring_heat_ingress", "blocked_impedance"]


@dataclass
class EngineConfig:
    fingerprint: dict | None = None   # from virtual_clone.learn_fingerprint (optional)
    du: DilutionUnit = field(default_factory=DilutionUnit)
    realistic: bool = True            # use realistic_faults physics (quench transient,
                                      # overlapping thermal faults, pressure/flow signatures,
                                      # sensor imperfections). See docs/FINDINGS.md #1-3.
    imperfections: bool = True        # layer sensor noise/dropouts/railing (realistic only)


def base_temps(loads_w: np.ndarray, du: DilutionUnit) -> np.ndarray:
    """Steady stage temperatures [K] from heat loads via the real T^2 floor.
    Upper stages (50K, 4K) are near-fixed (pulse tube); Still/CP/MXC from dilution."""
    T = np.empty(5)
    T[0] = 45.0 + loads_w[0] * 0.1        # 50K plate: weak load dependence
    T[1] = 3.8 + loads_w[1] * 0.5         # 4K plate
    T[2] = max(0.7, du.T_still_at_load(loads_w[2] + du.Q_still_nominal_W) * 0.7)
    T[3] = du.T_cp_at_load(loads_w[3] + 3e-6)   # cold plate, T^2
    T[4] = du.T_MXC_at_load(loads_w[4])         # mixing chamber, T^2 floor
    return T


def _fault_loads(fault_class: str, sev: float, active: float) -> np.ndarray:
    loads = NOMINAL_LOADS_W.copy()
    s = sev * active
    if fault_class == "heat_load_spike":
        loads[4] += 30e-6 * s
    elif fault_class == "helium_leak":
        loads[[2, 3, 4]] *= (1 + 4.0 * s)
    elif fault_class == "magnet_quench":
        loads[1] += 0.4 * s
    elif fault_class == "wiring_heat_ingress":
        loads[[2, 3, 4]] += np.array([5e-3, 20e-6, 6e-6]) * s
    elif fault_class == "blocked_impedance":
        loads[[2, 4]] *= (1 + 5.0 * s)
    return loads


@dataclass
class Scenario:
    fault_class: str
    severity: float = 0.0
    onset_frac: float = 0.4


def simulate(scenario: Scenario, cfg: EngineConfig, hours: float = 6.0,
             dt_min: float = 5.0, seed: int = 0) -> dict:
    """Generate a telemetry window: real T^2 physics mean + real-fridge fingerprint
    fluctuations + physics-grounded fault. Returns FRTMS-schema columns.

    When cfg.realistic (default), fault dynamics come from realistic_faults: the sharp
    magnet-quench transient, thermal faults that OVERLAP on temperature but separate on
    flow/pressure, and (optionally) sensor imperfections — the docs/FINDINGS.md #1-3 fix.
    The real BlueFors fingerprint noise is still layered on top when provided."""
    if getattr(cfg, "realistic", False):
        return _simulate_realistic(scenario, cfg, hours, dt_min, seed)
    return _simulate_clean(scenario, cfg, hours, dt_min, seed)


def _simulate_realistic(scenario: Scenario, cfg: EngineConfig, hours: float,
                        dt_min: float, seed: int) -> dict:
    """Realistic path: realistic_faults physics mean (sharp quench, overlapping thermal
    faults, flow/pressure signatures) + real BlueFors fingerprint fluctuations on the
    cold stages + sensor imperfections. Imported lazily to avoid a circular import."""
    from . import realistic_faults as RF

    # 1) realistic physics + flow/pressure signatures, WITHOUT imperfections yet
    #    (we add the fingerprint noise first, then imperfections last).
    cols = RF.simulate_realistic(scenario, cfg, hours=hours, dt_min=dt_min,
                                 seed=seed, imperfections=False)

    # 2) real-fridge fingerprint fluctuations on the physics mean (same mapping as the
    #    clean path): multiply each stage by its measured relative fluctuation.
    if cfg.fingerprint is not None:
        fp = cfg.fingerprint
        n = len(cols["t_s"])
        clone = VC.generate(fp, n, seed=seed)["series"]
        stages = fp["stages"]
        for ci, st in enumerate(stages):
            if st in STAGES:
                si = STAGES.index(st)
                rel = (clone[:, ci] - clone[:, ci].mean()) / (fp["median"][ci] + 1e-18)
                cols[f"temp{si+1}_T"] = cols[f"temp{si+1}_T"] * (1 + rel)
        if "MXC" in stages:  # CP borrows MXC's relative fluctuation (no CP channel in sample)
            ci = stages.index("MXC")
            rel = (clone[:, ci] - clone[:, ci].mean()) / (fp["median"][ci] + 1e-18)
            cols["temp4_T"] = cols["temp4_T"] * (1 + rel)

    # 3) sensor imperfections last (noise floor, calibration, dropouts, railing)
    if getattr(cfg, "imperfections", True):
        cols = RF.add_sensor_imperfections(cols, seed=seed)
    return cols


def _simulate_clean(scenario: Scenario, cfg: EngineConfig, hours: float = 6.0,
                    dt_min: float = 5.0, seed: int = 0) -> dict:
    """Legacy clean path: real T^2 physics mean + fingerprint fluctuations + the simple
    steady-load faults. Kept for the ablation (cfg.realistic=False) and reproducibility."""
    rng = np.random.default_rng(seed)
    n = max(int(hours * 60 / dt_min), 2)
    t_s = np.arange(n) * dt_min * 60.0
    onset = scenario.onset_frac * t_s[-1]

    # 1) physics mean trajectory
    mean_T = np.zeros((n, 5))
    for k in range(n):
        active = 0.0 if t_s[k] <= onset else min(1.0, (t_s[k] - onset) / (0.3 * t_s[-1] + 1))
        mean_T[k] = base_temps(_fault_loads(scenario.fault_class, scenario.severity, active), cfg.du)

    # 2) fluctuations: real fridge fingerprint if provided, else modest noise
    if cfg.fingerprint is not None:
        fp = cfg.fingerprint
        # map clone's 4 stages (50K,4K,Still,MXC) onto our 5; CP gets MXC-like character
        clone = VC.generate(fp, n, seed=seed)["series"]  # [n, 4] centered on real medians
        stages = fp["stages"]
        # use the clone's *fluctuation* (subtract its mean) as multiplicative-ish noise
        for ci, st in enumerate(stages):
            if st in STAGES:
                si = STAGES.index(st)
                fluct = clone[:, ci] - clone[:, ci].mean()
                # scale fluctuation to the current mean (relative noise preserved)
                rel = fluct / (fp["median"][ci] + 1e-18)
                mean_T[:, si] = mean_T[:, si] * (1 + rel)
        # CP: borrow MXC's relative fluctuation
        if "MXC" in stages:
            ci = stages.index("MXC")
            rel = (clone[:, ci] - clone[:, ci].mean()) / (fp["median"][ci] + 1e-18)
            mean_T[:, 3] = mean_T[:, 3] * (1 + rel)
    else:
        mean_T = mean_T * (1 + 0.01 * rng.standard_normal(mean_T.shape))

    cols = {"t_s": t_s}
    for i in range(5):
        cols[f"temp{i+1}_T"] = mean_T[:, i]
    # magnet sensors track 4K
    cols["temp6_T"] = mean_T[:, 1] * (1 + 0.01 * rng.standard_normal(n)) + 0.05
    cols["temp7_T"] = cols["temp6_T"].copy()
    cols["temp8_T"] = np.full(n, 300.0)
    flow = np.full(n, 0.68)
    if scenario.fault_class == "blocked_impedance":
        flow = np.where(t_s > onset, 0.68 * (1 - 0.5 * scenario.severity), 0.68)
    cols["flowmeter"] = np.maximum(flow * (1 + 0.02 * rng.standard_normal(n)), 0)
    for j in range(6):
        cols[f"p{j+1}"] = np.full(n, [1.5e3, 0.35, 8, 1e3, 2e-2, 5e2][j])
    cols["cpa_status"] = np.ones(n, dtype=int)
    cols["turbo_status"] = np.ones(n, dtype=int)
    return cols


def sample_scenario(rng: np.random.Generator) -> Scenario:
    fc = rng.choice(FAULT_CLASSES)
    sev = 0.0 if fc == "normal" else float(rng.uniform(0.3, 1.0))
    return Scenario(fc, sev, float(rng.uniform(0.3, 0.6)))
