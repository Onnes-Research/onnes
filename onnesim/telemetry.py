"""
OnnesSim telemetry emitter — VERIFIED FRTMS / Bluefors LD400 schema.

Schema source (verified, primary):
  - FRTMS paper, arXiv:2602.05160 (Esposito, Greenstein, Speller), §II.2 "Database Structure"
  - Repo Speller-Laboratory/JHUFridgeMonitoring-public, FRTMS_funcs.py (read directly)

The JHU MySQL tables are:
  p1..p6           bellows pressures
  temp1..temp8     temperatures for the different channels, INCLUDING TWO FOR THE MAGNET
                   (+ spare empty readout channels on the real system)
  flowmeter        flow of the mix
  cpa_status       compressor pump parameters
  turbo_status     turbo pump parameters

We map our 5 physical stages -> temp1..temp5, magnet sensors -> temp6,temp7,
and temp8 as a spare/empty channel, to mirror the real "8 temperature channels
including two for the magnet, plus spares" description.

⚠️ VALUES for pressures/flow/status remain PLACEHOLDER surrogate signals; only the
SCHEMA (column names + shape) is verified. See PHYSICS_NOTES.md.
"""

from __future__ import annotations
import numpy as np

from . import constants as C
from .faults import FaultSpec

# Verified FRTMS column order.
TEMP_COLS = [f"temp{i}_T" for i in range(1, 9)]      # temp1_T .. temp8_T
PRESSURE_COLS = [f"p{i}" for i in range(1, 7)]        # p1 .. p6
STATUS_COLS = ["flowmeter", "cpa_status", "turbo_status"]


def emit(sim: dict, spec: FaultSpec, seed: int = 0) -> dict:
    """Return a dict of named telemetry channels (numpy arrays) in FRTMS schema."""
    rng = np.random.default_rng(seed + 12345)
    t = sim["t_s"]
    T = sim["T"]  # [n_steps, n_stages]
    n_steps = T.shape[0]

    def noisy(x, rel):
        return x * (1.0 + rel * rng.standard_normal(np.shape(x)))

    cols = {"t_s": t}

    # temp1..temp5  <- the five physical stages (50K,4K,still,coldplate,mxc)
    for i in range(C.N_STAGES):
        cols[f"temp{i+1}_T"] = noisy(T[:, i], C.TEMP_NOISE_REL)

    # temp6, temp7 <- two magnet sensors, thermalized to the 4K flange (index FOURK=1)
    four_k = T[:, 1]
    magnet_base = four_k + C.MAGNET_OFFSET_K
    if spec.fault_class == "magnet_quench":
        a = np.clip((t - spec.onset_s) / 1800.0, 0.0, 1.0)
        magnet_base = magnet_base + C.MAGNET_QUENCH_SPIKE_K * spec.severity * a
    cols["temp6_T"] = noisy(magnet_base, C.TEMP_NOISE_REL)
    cols["temp7_T"] = noisy(magnet_base, C.TEMP_NOISE_REL)

    # temp8 <- spare/empty readout channel (real system: railed/placeholder)
    cols["temp8_T"] = np.full(n_steps, C.SPARE_CHANNEL_K)

    # Flow: nominal, reduced after onset if the fault restricts circulation.
    flow = np.full(n_steps, C.NOMINAL_FLOW)
    if spec.flow_scale != 1.0:
        active = t > spec.onset_s
        flow = np.where(active, C.NOMINAL_FLOW * spec.flow_scale, flow)
    cols["flowmeter"] = np.maximum(noisy(flow, C.FLOW_NOISE_REL), 0.0)

    # Pressures p1..p6: nominal + noise (surrogate).
    for j in range(6):
        cols[f"p{j+1}"] = np.maximum(
            noisy(np.full(n_steps, C.NOMINAL_PRESSURE[j]), C.PRESSURE_NOISE_REL), 0.0)

    # Discrete pump status (1 = running once engaged). Real cols pack many params;
    # we expose a single status bit each as a surrogate.
    engaged = (t > 60.0).astype(int)
    cols["cpa_status"] = engaged.copy()      # compressor
    cols["turbo_status"] = engaged.copy()    # turbo pump

    # Sensor fault: corrupt one physical temp channel after onset (telemetry-only).
    if spec.fault_class == "sensor_fault":
        ch = f"temp{spec.target_stage + 1}_T"
        active = t > spec.onset_s
        cols[ch] = np.where(active, cols[ch][0] * (1.0 + 3 * spec.severity), cols[ch])

    return cols


def to_csv(cols: dict, path: str) -> None:
    names = list(cols.keys())
    data = np.column_stack([cols[k] for k in names])
    header = ",".join(names)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6g")
