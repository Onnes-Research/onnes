"""
CryoOpsBench evaluation harness.

Scores two systems on the labeled scenarios produced by generate_dataset.py:
  1. CryoAgent (Claude or stub backend) — reads a telemetry window, judges it.
  2. Threshold baseline — mimics the FRTMS state of the art (fixed cutoffs on
     the coldest stages + magnet). This is the honest "what exists today" bar
     the agent must beat.

Metrics reported:
  - detection: does the system flag fault-vs-normal correctly? (accuracy / P / R)
  - classification: of the correctly-detected faults, is the class right?

Usage:
    python scripts/evaluate.py --dataset outputs/dataset --backend auto
"""

from __future__ import annotations
import numpy as np

from . import constants as C


# --- FRTMS-style threshold baseline ---------------------------------------- #
def threshold_baseline(cols: dict) -> dict:
    """Fixed-cutoff detector, in the spirit of FRTMS Grafana threshold rules.

    Flags a fault if any monitored channel ends outside a nominal band.
    Returns the same verdict shape as the agent (minimal fields)."""
    end = {k: float(np.asarray(v)[-1]) for k, v in cols.items() if k != "t_s"}

    mxc = end.get("temp5_T", 0.0)
    still = end.get("temp3_T", 0.0)
    magnet = end.get("temp6_T", 0.0)
    four_k = end.get("temp2_T", 0.0)
    flow = end.get("flowmeter", C.NOMINAL_FLOW)

    detected, fclass = False, "normal"
    # thresholds are ~1.5-2x nominal targets (PLACEHOLDER, like the rest of v0)
    if magnet > (C.T_TARGET[1] + C.MAGNET_OFFSET_K) * 1.5 or four_k > C.T_TARGET[1] * 1.5:
        detected, fclass = True, "magnet_quench"
    elif flow < C.NOMINAL_FLOW * 0.7:
        detected, fclass = True, "blocked_impedance"
    elif mxc > C.T_TARGET[4] * 2.0:
        detected, fclass = True, "helium_leak"
    elif still > C.T_TARGET[2] * 2.0:
        detected, fclass = True, "heat_load_spike"
    return {"fault_detected": detected, "fault_class": fclass}


# --- engineered physics-rule baseline -------------------------------------- #
def engineered_baseline(cols: dict) -> dict:
    """A stronger "what a good cryogenic engineer would build first" baseline.

    Unlike ``threshold_baseline`` (fixed cutoffs on end-of-window *levels* only),
    this encodes the standard operator toolkit: start-vs-end **deltas**,
    **rate-of-change** (temperature slew and the dQ/dt proxy), and the sign of
    **flow / pressure** changes on the specific channels that physically separate
    the overlapping faults. It uses NO training labels — thresholds come from the
    real sensor-noise floors (flow ~2.3 %, pressure ~0.6 %, cold-stage T <1 %),
    so anything well outside the noise band is treated as a real deviation.

    Rule order follows the fault physics (see realistic_faults.pressure_flow_signature):
      magnet_quench      sharp 4K/coil excursion (high dQ/dt, transient coil spike)
                         + condenser p1 boil-off spike, flow blips UP.
      helium_leak        circulation flow DOWN + OVC gauge p5 UP (vacuum degrades).
      blocked_impedance  flow DOWN (steeper) + condenser p1 UP, p5 FLAT (opposite leak).
      heat_load_spike    MXC-localized warming, flow/pressure FLAT, step onset.
      wiring_heat_ingress cold+adjacent stages warm together, flow/pressure FLAT.
    Returns the same verdict shape as the agent."""
    t = np.asarray(cols.get("t_s", []), dtype=float)

    def _seg(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        k = max(2, n // 4)
        base = np.nanmedian(x[:k])
        cur = np.nanmedian(x[-k:])
        return float(base), float(cur), float(np.nanmax(x)), float(np.nanmin(x))

    def pct(name):  # relative start->end change (robust segment medians)
        if name not in cols:
            return 0.0
        b, c, _, _ = _seg(cols[name])
        return (c - b) / abs(b) if b else 0.0

    def peak_pct(name):  # relative start->peak change (for transient spikes)
        if name not in cols:
            return 0.0
        b, _, hi, _ = _seg(cols[name])
        return (hi - b) / abs(b) if b else 0.0

    def slew(name):  # max |dX/dt| in units per minute (rate-of-change)
        if name not in cols or len(t) < 2:
            return 0.0
        x = np.asarray(cols[name], dtype=float)
        return float(np.nanmax(np.abs(np.gradient(x, t)))) * 60.0

    mxc, cp, still = pct("temp5_T"), pct("temp4_T"), pct("temp3_T")
    four_k = pct("temp2_T")
    mag_peak = peak_pct("temp6_T")
    flow = pct("flowmeter")
    p1, p5 = pct("p1"), pct("p5")
    # dQ/dt on the 4 K flange: the sharp heat-deposition-rate signature of a quench.
    dqdt = slew("dQdt_4K_Kpermin") if "dQdt_4K_Kpermin" in cols else slew("temp2_T")

    warm = max(mxc, cp, still)
    # detection: any channel meaningfully outside its noise band (flow noise ~2.3%,
    # pressure ~0.6%, cold-stage T <1%; bands set at a few sigma above those floors)
    detected = (warm > 0.04 or abs(flow) > 0.07 or mag_peak > 0.15
                or four_k > 0.08 or abs(p1) > 0.05 or abs(p5) > 0.05)
    if not detected:
        return {"fault_detected": False, "fault_class": "normal"}

    # 1) magnet quench — transient coil/4K spike + condenser boil-off
    if mag_peak > 0.30 or (dqdt > 0.5 and p1 > 0.10):
        return {"fault_detected": True, "fault_class": "magnet_quench"}
    # 2) helium leak — flow down + OVC (p5) up (the OVC rise is the leak signature)
    if p5 > 0.08 and flow < -0.04:
        return {"fault_detected": True, "fault_class": "helium_leak"}
    # 3) blocked impedance — flow down + condenser (p1) up, p5 flat
    if flow < -0.07 and p1 > 0.04:
        return {"fault_detected": True, "fault_class": "blocked_impedance"}
    # 4/5) thermal fault with flat flow/pressure: split by whether CP warms with MXC
    if warm > 0.04 and abs(flow) < 0.07:
        if cp > 0.03 and cp > 0.5 * max(mxc, 1e-9):
            return {"fault_detected": True, "fault_class": "wiring_heat_ingress"}
        return {"fault_detected": True, "fault_class": "heat_load_spike"}
    # detected but no clean signature match -> most common thermal fault
    return {"fault_detected": True, "fault_class": "heat_load_spike"}


# --- scoring --------------------------------------------------------------- #
def score(rows: list) -> dict:
    """rows: list of {truth_class, pred_detected, pred_class}."""
    tp = fp = tn = fn = 0
    class_correct = class_total = 0
    for r in rows:
        is_fault = r["truth_class"] != "normal"
        pred = r["pred_detected"]
        if is_fault and pred:
            tp += 1
            class_total += 1
            if r["pred_class"] == r["truth_class"]:
                class_correct += 1
        elif is_fault and not pred:
            fn += 1
        elif not is_fault and pred:
            fp += 1
        else:
            tn += 1
    n = max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": n,
        "detect_accuracy": round((tp + tn) / n, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
        "class_accuracy_on_detected": round(class_correct / max(class_total, 1), 3),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
