"""
bluefors_data.py — loader for REAL BlueFors dilution-fridge logs.

Source (VERIFIED, on disk): github.com/larchen/cryometrics, tests/logs/ — real
BlueFors log files from 6 named fridges (blizzard, oxicold, sisyphus, snowball,
snowflake, vericold). Standard BlueFors per-channel log format:
    CH<n> T <date>.log    -> "dd-mm-yy,HH:MM:SS,<temperature K>" per line
    Flowmeter <date>.log  -> "...,<flow mmol/s>"
    Status_<date>.log     -> many gas-handling/pump fields

Channel convention (BlueFors): CH1=50K, CH2=4K, CH5=Still, CH6=MXC (mixing chamber).
This is the closest REAL analog to OnnesSim's target: an actual dilution fridge at
~11 mK base. Second independent real fridge (after Leeds) for validation.

⚠️ LICENSE: the cryometrics repo states NO license (default copyright). Fine for
private evaluation; confirm with the author before any redistribution. Data is
git-ignored here.
"""
from __future__ import annotations
import glob
import os
import numpy as np

# BlueFors channel -> our stage name
CH_TO_STAGE = {"CH1": "50K", "CH2": "4K", "CH5": "Still", "CH6": "MXC"}


def load_channel(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one BlueFors 'CH<n> T <date>.log' or 'Flowmeter ...' file.
    Returns (time_seconds_from_start, values)."""
    ts, vals = [], []
    for line in open(path, encoding="latin-1", errors="replace"):
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 3:
            continue
        try:
            vals.append(float(parts[-1]))
        except ValueError:
            continue
        # date dd-mm-yy , time HH:MM:SS
        d, t = parts[0], parts[1]
        try:
            hh, mm, ss = (int(x) for x in t.split(":"))
            ts.append(hh * 3600 + mm * 60 + ss)
        except ValueError:
            ts.append(len(ts) * 60.0)  # fallback: assume 60s cadence
    t = np.array(ts, dtype=float)
    if len(t):
        t = t - t[0]
        # unwrap midnight rollover
        for i in range(1, len(t)):
            if t[i] < t[i - 1]:
                t[i:] += 24 * 3600
    return t, np.array(vals, dtype=float)


def load_fridge_day(log_dir: str) -> dict:
    """Load all stage temperatures + flow for one fridge/day directory.
    Returns {stage: (t, T)} plus 'flow' if present."""
    out = {}
    for ch, stage in CH_TO_STAGE.items():
        fs = glob.glob(os.path.join(log_dir, f"{ch} T*.log"))
        if fs:
            out[stage] = load_channel(fs[0])
    fs = glob.glob(os.path.join(log_dir, "Flowmeter*.log"))
    if fs:
        out["flow"] = load_channel(fs[0])
    return out


def base_temps(log_dir: str) -> dict:
    """Median (base) temperature per stage for a fridge/day — for validation."""
    data = load_fridge_day(log_dir)
    return {st: float(np.median(v[1])) for st, v in data.items()
            if st != "flow" and len(v[1])}
