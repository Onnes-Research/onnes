"""
realistic_faults.py — physically-realistic fault injection for the cryo engine.

WHY THIS FILE EXISTS (docs/FINDINGS.md)
--------------------------------------
The ML classifier scores F1 0.997 on `cryo_engine.py` output — a RED FLAG, not a
trophy. The synthetic faults are too cleanly separable, so models won't transfer to
real fridges. Two concrete defects were found:

  * `magnet_quench` is physically WRONG. At sev 0.9 the unedited engine warms the 4 K
    stage only +6 % (3.97 -> 4.21 K) as a slow ramp (max |dT/dt| = 0.026 K/min). A
    real quench is a SUDDEN, LARGE, TRANSIENT multi-Kelvin spike with a sharp dQ/dt.
    Live LLM agents correctly called this "wiring_heat_ingress" — they out-reasoned
    the fault model.
  * `helium_leak`, `blocked_impedance` and `wiring_heat_ingress` are separable from
    temperature alone, when in real hardware they OVERLAP (all warm the cold stages)
    and are told apart only by flow / pressure / dP-dt / onset dynamics.

This module fixes the PHYSICS of the faults without touching the engine. It provides
drop-in replacement fault-load + transient functions, a self-contained realistic
generator, and a sensor-imperfection layer that also operates on the engine's output.

PHYSICS GROUNDING (mechanism cited per fault, magnitudes sanity-checked vs real data)
------------------------------------------------------------------------------------
  magnet_quench        A superconducting solenoid (9 T, thermalized to the 4 K flange
                       per FRTMS arXiv:2602.05160) loses superconductivity; its stored
                       magnetic energy E = 1/2 L I^2 dumps into the windings as ohmic
                       heat over seconds. The 4 K flange sees a fast heat PULSE ->
                       multi-Kelvin spike, then the pulse tube (PT415: 1.5 W @ 4.2 K)
                       re-cools it over tens of minutes -> PARTIAL recovery. Sharp
                       dQ/dt is the signature (Pobell, "Matter and Methods at Low
                       Temperatures"; Iwasa, "Case Studies in Superconducting Magnets").
  helium_leak          Loss of 3He/4He mixture or a leak into the OVC. Circulation
                       (flow) DROPS, the insulating-vacuum gauge RISES (vacuum
                       degrades), cold-stage cooling fades -> Still/CP/MXC warm SLOWLY.
  blocked_impedance    The main flow impedance (condenser->still capillary) partially
                       plugs with solid air/ice. Flow DROPS while the condenser-side
                       pressure RISES (upstream buildup) — the OPPOSITE dP/dt sign to a
                       leak. Starves the MXC -> Still/MXC warm.
  wiring_heat_ingress  Poorly heat-sunk wiring conducts room-side heat into the cold
                       stages: a near-constant parasitic load on Still/CP/MXC. Flow and
                       pressure stay FLAT — the discriminator vs leak/block.
  heat_load_spike      A localized load step at the MXC (measurement pulse / sample
                       heater). Sharp onset, MXC-localized, no flow/pressure signature.

The three thermal faults are TUNED TO OVERLAP on temperature (same stages, comparable
CP/MXC rise) and are separated only by the flow + pressure + dP/dt + onset channels —
so a temperature-only classifier is genuinely confused, as on real hardware.

REAL-DATA ANCHORS (measured on disk this session; see leeds/bluefors/ornl loaders)
  * BlueFors 11 mK fridge sample — per-stage relative noise floor used below:
      50K 1.6 %, 4K 0.10 %, Still 0.02 %, MXC 0.74 % (sample-to-sample); flow 2.3 %.
  * ORNL SNS cryoplant (real T+P+flow+valves) — steady pressures move at 0.2-1 % rel,
    flows 0.6-2.3 %; a real thermal transient (TI_6102) slews 2.2 K/min. A quench must
    be faster/larger than that; the leak/block pressure moves here are of order a few
    to tens of %, comfortably above the real 1 % noise floor yet physically modest.

RULES OBSERVED: this module edits NOTHING else. It imports `cryo_engine.base_temps`
and `dilution_cooling.DilutionUnit` (the real T^2 floor) as read-only building blocks.
Every magnitude below is a modelling choice; the __main__ demo REPORTS what it actually
produces (no number here is asserted as a validation result).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import cryo_engine as CE
from .dilution_cooling import DilutionUnit

STAGES = CE.STAGES                       # ["50K", "4K", "Still", "CP", "MXC"]
NOMINAL_LOADS_W = CE.NOMINAL_LOADS_W     # [15.0, 0.30, 8.0e-4, 30e-6, 5.0e-6]
FAULT_CLASSES = CE.FAULT_CLASSES
NOMINAL_FLOW = 0.68                      # mmol/s (engine convention; real BlueFors ~0.70)
NOMINAL_P = np.array([1.5e3, 0.35, 8.0, 1.0e3, 2.0e-2, 5.0e2])  # p1..p6 [mbar]

# Per-stage RELATIVE noise floor, measured on the real BlueFors 11 mK sample this
# session (std of first differences / median), rounded up slightly to keep the
# benchmark honest. CP has no real channel in the sample -> conservative estimate.
REAL_NOISE_REL = {"50K": 0.016, "4K": 0.0015, "Still": 0.0006, "CP": 0.005, "MXC": 0.0074}
# Additive noise floor [K] so relative noise never collapses to zero on the coldest
# stages (a real thermometer has an absolute resolution limit).
ADDITIVE_FLOOR_K = {"50K": 0.02, "4K": 3e-3, "Still": 2e-4, "CP": 3e-4, "MXC": 3e-5}
FLOW_NOISE_REL = 0.023      # real BlueFors flowmeter
P_NOISE_REL = 0.006         # real ORNL pressure gauges (~0.2-1 %); use 0.6 %


# --------------------------------------------------------------------------------------
# 1. Onset shapes — real faults are NOT all identical monotonic ramps.
# --------------------------------------------------------------------------------------
def onset_shape(kind: str, t_s: np.ndarray, onset_s: float, window_s: float) -> np.ndarray:
    """Activation s(t) in [0, 1] describing how a fault turns on.

    kind: 'slow'   exponential approach, tau = 0.5*window   (helium_leak: gradual)
          'medium' exponential approach, tau = 0.15*window  (block / wiring: minutes)
          'step'   near-instant with ~2-sample smoothing    (heat_load_spike)
          'spike'  difference-of-exponentials, fast rise + slow decay (magnet_quench)
    Returns an array the length of t_s.
    """
    d = t_s - onset_s
    on = d >= 0
    s = np.zeros_like(t_s, dtype=float)
    if kind == "slow":
        s[on] = 1.0 - np.exp(-d[on] / (0.5 * window_s + 1e-9))
    elif kind == "medium":
        s[on] = 1.0 - np.exp(-d[on] / (0.15 * window_s + 1e-9))
    elif kind == "step":
        s[on] = 1.0 - np.exp(-d[on] / (2.0 * _dt_guess(t_s)))
    elif kind == "spike":
        s = _diff_exp(d, tr_s=48.0, td_s=25.0 * 60.0)   # normalized to unit peak
    else:
        raise ValueError(f"unknown onset kind {kind!r}")
    return np.clip(s, 0.0, 1.0 if kind != "spike" else None)


def _dt_guess(t_s: np.ndarray) -> float:
    return float(np.median(np.diff(t_s))) if len(t_s) > 1 else 60.0


def _diff_exp(delta_s: np.ndarray, tr_s: float, td_s: float) -> np.ndarray:
    """Difference-of-exponentials pulse, normalized to unit peak.
    Fast rise (tr_s), slow decay (td_s): the canonical quench-flange response."""
    g = np.where(delta_s >= 0,
                 np.exp(-np.maximum(delta_s, 0.0) / td_s) - np.exp(-np.maximum(delta_s, 0.0) / tr_s),
                 0.0)
    t_pk = (tr_s * td_s / (td_s - tr_s)) * np.log(td_s / tr_s)
    g_pk = np.exp(-t_pk / td_s) - np.exp(-t_pk / tr_s)
    return g / max(g_pk, 1e-12)


# --------------------------------------------------------------------------------------
# 2. Fault heat loads — DROP-IN replacement for cryo_engine._fault_loads.
#    Tuned so helium_leak / blocked_impedance / wiring_heat_ingress OVERLAP on the
#    cold-stage temperatures (same stages, comparable rise). They are pulled apart
#    only by pressure_flow_signature() + onset shape + dP/dt sign.
# --------------------------------------------------------------------------------------
def fault_loads(fault_class: str, sev: float, active: float) -> np.ndarray:
    """Steady per-stage heat loads [W] for a fault at activation `active` in [0,1].

    Drop-in for cryo_engine._fault_loads(). NOTE: magnet_quench carries NO steady load
    here — its whole excursion is the transient in quench_transient() (a heat pulse,
    not a sustained load), which is the physical correction FINDINGS.md #2 demands.
    """
    loads = NOMINAL_LOADS_W.copy()
    s = float(sev) * float(active)
    if s <= 0.0 or fault_class in ("normal", "magnet_quench"):
        return loads

    # Common cold-stage envelope shared by the three thermal faults => temperature
    # OVERLAP. Indices: [50K, 4K, Still, CP, MXC].
    if fault_class == "helium_leak":
        # Mixture loss: cooling fades most at MXC/CP; Still barely moves.
        loads[4] += NOMINAL_LOADS_W[4] * (6.0 * s)      # MXC
        loads[3] += NOMINAL_LOADS_W[3] * (5.0 * s)      # CP
        loads[2] += 2.0e-3 * s                          # Still (small)
    elif fault_class == "blocked_impedance":
        # Flow starves the MXC and backs up at the still: MXC/CP warm like a leak,
        # Still warms a touch more.
        loads[4] += NOMINAL_LOADS_W[4] * (6.2 * s)      # MXC (overlaps leak)
        loads[3] += NOMINAL_LOADS_W[3] * (4.5 * s)      # CP  (overlaps leak)
        loads[2] += 6.0e-3 * s                          # Still (more than leak)
    elif fault_class == "wiring_heat_ingress":
        # Conducted parasitic reaching ALL cold stages; comparable MXC/CP rise.
        loads[4] += NOMINAL_LOADS_W[4] * (5.8 * s)      # MXC (overlaps leak/block)
        loads[3] += NOMINAL_LOADS_W[3] * (5.2 * s)      # CP  (overlaps leak/block)
        loads[2] += 5.0e-3 * s                          # Still
    elif fault_class == "heat_load_spike":
        # Localized MXC load step; leaves CP/Still alone (distinct footprint).
        loads[4] += NOMINAL_LOADS_W[4] * (7.0 * s)      # MXC only
    return loads


# --------------------------------------------------------------------------------------
# 3. Magnet-quench transient — the headline fix: sharp multi-Kelvin spike + recovery.
# --------------------------------------------------------------------------------------
def quench_transient(t_s: np.ndarray, onset_s: float, sev: float) -> dict:
    """Additive temperature excursions [K] from a magnet quench.

    Mechanism: E = 1/2 L I^2 stored in the 9 T solenoid dumps into the windings on
    quench (seconds), heating the 4 K flange the magnet is thermalized to; the pulse
    tube then re-cools over ~tens of minutes -> PARTIAL recovery (settles slightly
    above base while the now-normal magnet + boiled-off gas add residual load).

    Returns dict of arrays: dT_4K, dT_magnet (sensor on the coil, spikes harder),
    dT_cold (small delayed bleed into Still/CP/MXC), and dQ_arb (a dQ/dt proxy:
    the analytic time-derivative of the pulse, so the sharp heat-deposition rate is a
    first-class channel). All zero before onset.
    """
    s = float(np.clip(sev, 0.0, 1.0))
    d = t_s - onset_s
    tr_s, td_s = 48.0, 25.0 * 60.0                  # 0.8 min rise, 25 min recover
    pulse = _diff_exp(d, tr_s, td_s)                 # unit-peak spike
    settle = np.where(d >= 0, 1.0 - np.exp(-np.maximum(d, 0.0) / (3.0 * tr_s)), 0.0)

    dT_4K = 14.0 * s * pulse + 0.35 * s * settle     # peak ~multi-K, resid partial recovery
    dT_magnet = 55.0 * s * pulse + 1.20 * s * settle  # coil sensor spikes hardest
    # small, briefly-delayed bleed into the cold platform (overlaps thermal faults
    # transiently — but the sharp 4K/magnet spike is the discriminator)
    bleed = _diff_exp(d - 60.0, tr_s * 4, td_s)
    dT_cold = np.stack([0.020 * s * bleed,           # Still
                        0.006 * s * bleed,           # CP
                        0.010 * s * bleed], axis=-1) # MXC
    # dQ/dt proxy: derivative of the pulse (K/s), sharp positive spike at onset
    dQ_arb = np.gradient(dT_4K, t_s)
    return {"dT_4K": dT_4K, "dT_magnet": dT_magnet, "dT_cold": dT_cold, "dQ_arb": dQ_arb}


# --------------------------------------------------------------------------------------
# 4. Pressure / flow signatures — the distinguishing channels for the overlapping faults.
#    Magnitudes kept in the "few % to tens of %" band that sits above the real ~1 %
#    gauge noise (ORNL) yet stays physically modest.
# --------------------------------------------------------------------------------------
def pressure_flow_signature(fault_class: str, sev: float, t_s: np.ndarray,
                            onset_s: float, window_s: float) -> dict:
    """Multiplicative envelopes for flow and p1..p6 vs time (nominal = 1.0).

    Sign conventions grounded in the mechanism:
      helium_leak       flow DOWN (slow), OVC gauge p5 UP (vacuum degrades),
                        still line p2 DOWN.               -> dP5/dt > 0
      blocked_impedance flow DOWN (steeper), condenser p1 UP (upstream buildup),
                        still line p2 DOWN, p3 UP.        -> dP1/dt > 0  (opposite leak)
      wiring_heat_ingress   flow + all pressures FLAT     -> the null signature
      heat_load_spike   flow + pressures FLAT
      magnet_quench     fast flow blip UP + p1 pressure spike (He boil-off), recover
    """
    flow = np.ones_like(t_s, dtype=float)
    p = np.ones((len(t_s), 6), dtype=float)
    s = float(sev)
    if s <= 0.0 or fault_class in ("normal", "wiring_heat_ingress", "heat_load_spike"):
        return {"flow_mult": flow, "p_mult": p}

    if fault_class == "helium_leak":
        a = onset_shape("slow", t_s, onset_s, window_s)
        flow *= 1.0 - 0.35 * s * a
        p[:, 4] *= 1.0 + 3.0 * s * a       # p5 OVC vacuum RISES (leak into OVC)
        p[:, 1] *= 1.0 - 0.20 * s * a      # p2 still line falls (inventory loss)
        p[:, 0] *= 1.0 - 0.05 * s * a      # p1 condenser slightly down
    elif fault_class == "blocked_impedance":
        a = onset_shape("medium", t_s, onset_s, window_s)
        flow *= 1.0 - 0.50 * s * a         # steeper flow drop
        p[:, 0] *= 1.0 + 0.15 * s * a      # p1 condenser RISES (backs up upstream)
        p[:, 2] *= 1.0 + 0.20 * s * a      # p3 up
        p[:, 1] *= 1.0 - 0.25 * s * a      # p2 still line starves
        # p5 (OVC) stays flat -> tells block apart from leak
    elif fault_class == "magnet_quench":
        g = _diff_exp(t_s - onset_s, tr_s=48.0, td_s=25.0 * 60.0)
        flow *= 1.0 + 0.15 * s * g         # brief circulation blip from gas expansion
        p[:, 0] *= 1.0 + 0.40 * s * g      # p1 pressure spike (He boil-off), sharp
    return {"flow_mult": flow, "p_mult": p}


# --------------------------------------------------------------------------------------
# 5. Sensor imperfections — noise floors, calibration offsets, dropouts/NaN, railing.
#    Operates IN PLACE on any FRTMS-schema cols dict (engine output or ours).
# --------------------------------------------------------------------------------------
def add_sensor_imperfections(cols: dict, seed: int = 0, drop_prob: float = 0.002,
                             calibrate: bool = True, noise: bool = True,
                             dropouts: bool = True) -> dict:
    """Layer realistic sensor defects onto a telemetry dict (modifies a copy).

    * calibration: per-RUN fixed gain + additive offset per channel (thermometry is
      never perfectly calibrated) -> shifts class means run-to-run so absolute
      thresholds don't cleanly separate classes.
    * noise: per-channel relative floor from the REAL BlueFors fridge + an absolute
      resolution floor on the coldest stages.
    * dropouts: sparse single-sample NaN and short bursts (sensor read failures); the
      magnet channels additionally RAIL to 300 K when driven hot (over-range), matching
      the FRTMS spare/magnet-channel behavior noted in docs/PHYSICS_REVIEW.md.
    """
    rng = np.random.default_rng(seed + 9973)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cols.items()}
    n = len(out["t_s"])

    temp_map = {"temp1_T": "50K", "temp2_T": "4K", "temp3_T": "Still",
                "temp4_T": "CP", "temp5_T": "MXC", "temp6_T": "4K", "temp7_T": "4K"}
    for key, stage in temp_map.items():
        if key not in out:
            continue
        x = out[key].astype(float)
        if calibrate:
            gain = 1.0 + rng.normal(0.0, 0.008)             # ~0.8 % gain error
            offset = rng.normal(0.0, ADDITIVE_FLOOR_K[stage])
            x = x * gain + offset
        if noise:
            rel = REAL_NOISE_REL[stage]
            x = x * (1.0 + rel * rng.standard_normal(n)) + \
                ADDITIVE_FLOOR_K[stage] * rng.standard_normal(n)
        out[key] = x

    if "flowmeter" in out:
        f = out["flowmeter"].astype(float)
        if calibrate:
            f = f * (1.0 + rng.normal(0.0, 0.02))
        if noise:
            f = f * (1.0 + FLOW_NOISE_REL * rng.standard_normal(n))
        out["flowmeter"] = np.maximum(f, 0.0)

    for j in range(1, 7):
        key = f"p{j}"
        if key not in out:
            continue
        pj = out[key].astype(float)
        if calibrate:
            pj = pj * (1.0 + rng.normal(0.0, 0.01))
        if noise:
            pj = pj * (1.0 + P_NOISE_REL * rng.standard_normal(n))
        out[key] = pj

    if dropouts:
        chans = [k for k in out if (k.startswith("temp") or k in ("flowmeter",)
                 or (k.startswith("p") and k[1:].isdigit()))]
        for key in chans:
            x = out[key].astype(float)
            mask = rng.random(n) < drop_prob
            # occasional 2-4 sample burst dropout
            if rng.random() < 0.25 and n > 6:
                b = rng.integers(0, n - 4)
                mask[b:b + rng.integers(2, 5)] = True
            x[mask] = np.nan
            out[key] = x
        # magnet channels rail to 300 K (over-range) when driven very hot by a quench
        for key in ("temp6_T", "temp7_T"):
            if key in out:
                x = out[key]
                hot = np.nan_to_num(x, nan=0.0) > 40.0
                rail = hot & (rng.random(n) < 0.3)
                x[rail] = 300.0
                out[key] = x
    return out


# --------------------------------------------------------------------------------------
# 6. Self-contained realistic generator (uses the engine's real T^2 base_temps).
# --------------------------------------------------------------------------------------
def simulate_realistic(scenario: CE.Scenario, cfg: CE.EngineConfig | None = None,
                       hours: float = 6.0, dt_min: float = 1.0, seed: int = 0,
                       imperfections: bool = True) -> dict:
    """Generate a realistic telemetry window: real T^2 physics mean + realistic fault
    dynamics (sharp quench transient, overlapping thermal loads, flow/pressure
    signatures) + optional sensor imperfections. Returns the FRTMS schema.

    With imperfections=False the physics is deterministic (no RNG) — use it to read
    exact base/peak values and onset sharpness (the demo does this for before/after).
    """
    cfg = cfg or CE.EngineConfig()
    du = cfg.du if cfg.du is not None else DilutionUnit()
    n = max(int(hours * 60 / dt_min), 2)
    t_s = np.arange(n) * dt_min * 60.0
    onset_s = scenario.onset_frac * t_s[-1]
    window_s = t_s[-1]
    fc, sev = scenario.fault_class, scenario.severity

    onset_kind = {"helium_leak": "slow", "blocked_impedance": "medium",
                  "wiring_heat_ingress": "medium", "heat_load_spike": "step",
                  "magnet_quench": "spike", "normal": "slow"}.get(fc, "slow")
    a = onset_shape(onset_kind, t_s, onset_s, window_s)
    if onset_kind == "spike":                       # loads use a plain ramp; spike is transient
        a = onset_shape("medium", t_s, onset_s, window_s)

    # 1) physics mean trajectory from the real T^2 floor
    mean_T = np.empty((n, 5))
    for k in range(n):
        mean_T[k] = CE.base_temps(fault_loads(fc, sev, float(a[k])), du)

    # 2) magnet-quench transient (sharp spike + partial recovery)
    q = quench_transient(t_s, onset_s, sev) if fc == "magnet_quench" else None
    if q is not None:
        mean_T[:, 1] += q["dT_4K"]
        mean_T[:, 2] += q["dT_cold"][:, 0]
        mean_T[:, 3] += q["dT_cold"][:, 1]
        mean_T[:, 4] += q["dT_cold"][:, 2]

    cols = {"t_s": t_s}
    for i in range(5):
        cols[f"temp{i+1}_T"] = mean_T[:, i].copy()
    # magnet sensors track 4 K + offset; add the (larger) quench excursion on the coil
    mag = mean_T[:, 1].copy() + 0.05
    if q is not None:
        mag = mag - q["dT_4K"] + q["dT_magnet"]     # replace 4K bleed with coil spike
    cols["temp6_T"] = mag
    cols["temp7_T"] = mag.copy()
    cols["temp8_T"] = np.full(n, 300.0)             # spare channel railed (real behavior)

    # 3) flow + pressure signatures
    sig = pressure_flow_signature(fc, sev, t_s, onset_s, window_s)
    cols["flowmeter"] = np.maximum(NOMINAL_FLOW * sig["flow_mult"], 0.0)
    for j in range(6):
        cols[f"p{j+1}"] = NOMINAL_P[j] * sig["p_mult"][:, j]
    cols["cpa_status"] = np.ones(n, dtype=int)
    cols["turbo_status"] = np.ones(n, dtype=int)
    # dQ/dt proxy channel (K/min) — the sharp heat-deposition-rate signature
    cols["dQdt_4K_Kpermin"] = (np.gradient(cols["temp2_T"], t_s) * 60.0)

    if imperfections:
        cols = add_sensor_imperfections(cols, seed=seed)
    return cols


# --------------------------------------------------------------------------------------
# Demo helpers
# --------------------------------------------------------------------------------------
def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _max_slew_Kpermin(t_s: np.ndarray, x: np.ndarray) -> float:
    m = np.isfinite(x)
    if m.sum() < 2:
        return float("nan")
    return float(np.nanmax(np.abs(np.diff(x[m]) / (np.diff(t_s[m]) / 60.0))))


def _pre_post(t_s, x, onset_s):
    pre = _finite(x[t_s < onset_s])
    post = _finite(x[t_s >= onset_s])
    base = float(np.median(pre)) if len(pre) else float("nan")
    peak = float(post[np.nanargmax(np.abs(post - base))]) if len(post) else float("nan")
    settled = float(np.median(post[-max(3, len(post)//5):])) if len(post) else float("nan")
    return base, peak, settled


# --------------------------------------------------------------------------------------
# __main__ demo — prints, per fault class, what moves and how; PROVES the quench fix.
# --------------------------------------------------------------------------------------
def _demo():
    cfg = CE.EngineConfig()
    hours, dt_min, sev = 6.0, 1.0, 0.9

    print("=" * 90)
    print("REALISTIC FAULT DEMO  (severity 0.9, 6 h window, 1-min cadence)")
    print("Base physics = cryo_engine.base_temps (real 3He/4He T^2 floor). Values are")
    print("from imperfections=False runs (deterministic physics) unless noted.")
    print("=" * 90)

    # ---- magnet_quench BEFORE (unedited engine) vs AFTER (this module) ----
    print("\n[1] magnet_quench — BEFORE (cryo_engine) vs AFTER (realistic_faults)")
    sc = CE.Scenario("magnet_quench", severity=sev, onset_frac=0.4)
    before = CE.simulate(sc, cfg, hours=hours, dt_min=dt_min, seed=0)
    t_b = before["t_s"]; onset_b = 0.4 * t_b[-1]
    b_base, b_peak, _ = _pre_post(t_b, before["temp2_T"], onset_b)
    b_slew = _max_slew_Kpermin(t_b, before["temp2_T"])
    print(f"    BEFORE 4K: base {b_base:6.3f} K -> peak {b_peak:6.3f} K "
          f"(+{100*(b_peak-b_base)/b_base:4.1f} %)  max|dT/dt| {b_slew:.3f} K/min  [gentle ramp]")

    after = simulate_realistic(sc, cfg, hours=hours, dt_min=dt_min, seed=0, imperfections=False)
    t_a = after["t_s"]; onset_a = 0.4 * t_a[-1]
    a_base, a_peak, a_set = _pre_post(t_a, after["temp2_T"], onset_a)
    a_slew = _max_slew_Kpermin(t_a, after["temp2_T"])
    m_base, m_peak, _ = _pre_post(t_a, after["temp6_T"], onset_a)
    dq = after["dQdt_4K_Kpermin"]
    print(f"    AFTER  4K: base {a_base:6.3f} K -> peak {a_peak:6.3f} K "
          f"(+{100*(a_peak-a_base)/a_base:5.1f} %)  max|dT/dt| {a_slew:.3f} K/min  [SHARP spike]")
    print(f"    AFTER  4K settles to {a_set:6.3f} K  (PARTIAL recovery, not full)")
    print(f"    AFTER  magnet sensor: base {m_base:6.3f} K -> peak {m_peak:6.2f} K  "
          f"(coil spikes hardest)")
    print(f"    AFTER  dQ/dt proxy peak: {np.nanmax(dq):+.2f} K/min  (sharp heat dump)")
    print(f"    >>> peak 4K rise x{(a_peak-a_base)/(b_peak-b_base):.0f} larger, "
          f"slew x{a_slew/b_slew:.0f} sharper than before")

    # ---- per-class summary: what stages/channels move ----
    print("\n[2] Per-class footprint (base -> peak; flow & pressure sign; onset slew)")
    hdr = (f"    {'class':20s} {'Still':>13s} {'CP':>13s} {'MXC':>15s} "
           f"{'flow%':>7s} {'p1%':>6s} {'p5%':>6s} {'onset K/min':>11s}")
    print(hdr)
    for fc in FAULT_CLASSES:
        sc = CE.Scenario(fc, severity=(0.0 if fc == "normal" else sev), onset_frac=0.4)
        c = simulate_realistic(sc, cfg, hours=hours, dt_min=dt_min, seed=1, imperfections=False)
        t = c["t_s"]; on = 0.4 * t[-1]
        def mv(key):
            base, peak, _ = _pre_post(t, c[key], on)
            return base, peak
        sb, sp = mv("temp3_T"); cb, cp = mv("temp4_T"); mb, mp = mv("temp5_T")
        fb, fp = mv("flowmeter"); p1b, p1p = mv("p1"); p5b, p5p = mv("p5")
        dmxc = _max_slew_Kpermin(t, c["temp5_T"])
        dq4 = _max_slew_Kpermin(t, c["temp2_T"])
        slew = max(dmxc, dq4)
        fpct = 100 * (fp - fb) / fb if fb else 0.0
        p1pct = 100 * (p1p - p1b) / p1b if p1b else 0.0
        p5pct = 100 * (p5p - p5b) / p5b if p5b else 0.0
        print(f"    {fc:20s} "
              f"{sb*1e3:5.0f}->{sp*1e3:4.0f}mK {cb*1e3:5.0f}->{cp*1e3:4.0f}mK "
              f"{mb*1e3:6.1f}->{mp*1e3:5.1f}mK "
              f"{fpct:+6.1f} {p1pct:+5.1f} {p5pct:+5.1f} {slew:10.2f}")

    # ---- overlap proof: leak / block / wiring on temperature vs on P/flow ----
    print("\n[3] OVERLAP proof — leak / block / wiring warm the SAME cold stages")
    print("    (temperature alone is ambiguous; flow + p5(OVC) + p1(condenser) separate)")
    for fc in ("helium_leak", "blocked_impedance", "wiring_heat_ingress"):
        sc = CE.Scenario(fc, severity=sev, onset_frac=0.4)
        c = simulate_realistic(sc, cfg, hours=hours, dt_min=dt_min, seed=2, imperfections=False)
        t = c["t_s"]; on = 0.4 * t[-1]
        _, mp, _ = _pre_post(t, c["temp5_T"], on)      # MXC peak
        _, cp, _ = _pre_post(t, c["temp4_T"], on)      # CP peak
        fb, fp, _ = _pre_post(t, c["flowmeter"], on)
        _, p1p, _ = _pre_post(t, c["p1"], on)
        _, p5p, _ = _pre_post(t, c["p5"], on)
        flow_sign = "flat" if abs(fp/fb - 1) < 0.02 else ("DOWN" if fp < fb else "UP")
        p1_sign = "flat" if abs(p1p/NOMINAL_P[0]-1) < 0.02 else ("UP" if p1p > NOMINAL_P[0] else "DOWN")
        p5_sign = "flat" if abs(p5p/NOMINAL_P[4]-1) < 0.02 else ("UP" if p5p > NOMINAL_P[4] else "DOWN")
        print(f"    {fc:20s} MXC~{mp*1e3:4.0f}mK CP~{cp*1e3:3.0f}mK  |  "
              f"flow {flow_sign:4s}  p1(condenser) {p1_sign:4s}  p5(OVC) {p5_sign:4s}")
    print("    -> identical thermal footprint, DISTINCT flow/pressure fingerprint.")

    # ---- sensor imperfections summary ----
    print("\n[4] Sensor imperfections (imperfections=True)")
    sc = CE.Scenario("helium_leak", severity=sev, onset_frac=0.4)
    clean = simulate_realistic(sc, cfg, hours=hours, dt_min=dt_min, seed=3, imperfections=False)
    dirty = simulate_realistic(sc, cfg, hours=hours, dt_min=dt_min, seed=3, imperfections=True)
    nan_total = 0
    for key in dirty:
        v = dirty[key]
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            nan_total += int(np.sum(~np.isfinite(v)))
    # calibration offset on MXC (median shift on the pre-onset segment)
    t = clean["t_s"]; on = 0.4 * t[-1]
    seg = t < on
    mxc_shift = np.nanmedian(dirty["temp5_T"][seg]) - np.median(clean["temp5_T"][seg])
    print(f"    noise floors (real BlueFors): 50K {REAL_NOISE_REL['50K']*100:.1f}% "
          f"4K {REAL_NOISE_REL['4K']*100:.2f}% Still {REAL_NOISE_REL['Still']*100:.2f}% "
          f"MXC {REAL_NOISE_REL['MXC']*100:.2f}%  flow {FLOW_NOISE_REL*100:.1f}%")
    print(f"    total NaN dropouts injected across channels: {nan_total}")
    print(f"    per-run MXC calibration shift: {mxc_shift*1e3:+.3f} mK "
          f"(class means move run-to-run -> no trivial threshold)")
    # railing check on the quench magnet channel
    sc_q = CE.Scenario("magnet_quench", severity=sev, onset_frac=0.4)
    dq = simulate_realistic(sc_q, cfg, hours=hours, dt_min=dt_min, seed=4, imperfections=True)
    railed = int(np.sum(np.nan_to_num(dq["temp6_T"], nan=0.0) >= 300.0))
    print(f"    magnet channel railed to 300 K during quench: {railed} samples "
          f"(sensor over-range, matches FRTMS spare/magnet behavior)")
    print("=" * 90)


if __name__ == "__main__":
    _demo()
