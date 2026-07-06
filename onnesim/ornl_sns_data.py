"""
ornl_sns_data.py — loader for the REAL ORNL SNS cryogenic-moderator-system dataset.

Source (VERIFIED, downloaded via Globus by the user): ORNL, "Process and control
variables from the SNS's cryogenic moderator system", Maldonado/Winder/Ramuhalli/
Blokland (2024), DOI 10.13139/OLCF/2441156. 54 signals over 2021-10-20..21.
Citation required (see README.txt). DOE Office of Science user-facility data.

This is a REAL cryogenic HELIUM refrigeration plant (coldbox + turbine + compressor)
feeding an H2 moderator loop (3 heat exchangers). Signals: temperatures (K),
pressures (psig), mass flows (gps), valve positions (%), turbine speed (rpm), etc.

⚠️ SCOPE: this is a ~15-20 K neutron-source cryoplant, NOT a millikelvin dilution
fridge. It is the richest REAL multivariate cryo telemetry we have (T + P + flow +
valves + a real refrigeration turbine/compressor), and a real third cryo source for
the agent/ML to consume. Different scale from OnnesSim's dilution-fridge target.

File format: header lines starting '#', then 'YYYY-MM-DD HH:MM:SS.ffffff, value'.
Missing-data codes (per README): ' Disconnected', ' Write_Error', ' Archive_Off' -> NaN.
Signal meaning is in the filename; human labels are in README.txt's file list.
"""
from __future__ import annotations
import glob
import os
import numpy as np

ORNL_DIR = os.path.join("data", "real", "ornl_sns")
MISSING = {"disconnected", "write_error", "archive_off", "nan", ""}

# Signal-type prefix in the PV name -> kind
KIND = {"TI": "temperature_K", "PI": "pressure_psig", "PT": "pressure_psig",
        "DPI": "dpressure_psid", "FI": "flow_gps", "SI": "speed_rpm",
        "II": "current_W", "JI": "power_W", "ZI": "level"}


def _parse_value(s: str) -> float:
    s2 = s.strip().lower()
    if s2 in MISSING:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_signal(path: str) -> dict:
    """Load one ORNL signal CSV -> {pv, kind, t_s, values}. Times are seconds from start."""
    pv = ""
    ts, vals = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "Archiver ID" in line or ":" in line:
                pv = pv or line.lstrip("# ").split(",")[0].strip()
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        stamp, val = parts
        try:
            # 'YYYY-MM-DD HH:MM:SS.ffffff'
            hms = stamp.strip().split(" ")[1]
            h, m, s = hms.split(":")
            ts.append(int(h) * 3600 + int(m) * 60 + float(s))
        except (ValueError, IndexError):
            ts.append(len(ts) * 60.0)
        vals.append(_parse_value(val))
    t = np.array(ts, dtype=float)
    if len(t):
        t -= t[0]
        for i in range(1, len(t)):
            if t[i] < t[i - 1]:
                t[i:] += 24 * 3600
    # kind from filename PV: TGT_<sys>-IOC1-<KIND>_<num>
    base = os.path.basename(path)
    kind = "other"
    for k, v in KIND.items():
        if f"-{k}_" in base or f"-{k}-" in base:
            kind = v
            break
    return {"pv": pv or base.split("__")[0], "kind": kind,
            "t_s": t, "values": np.array(vals, dtype=float)}


def list_signals(kind: str | None = None) -> list[str]:
    """Paths to ORNL signal CSVs, optionally filtered by kind (e.g. 'temperature_K')."""
    paths = sorted(glob.glob(os.path.join(ORNL_DIR, "*.csv")))
    if kind is None:
        return paths
    return [p for p in paths if load_signal(p)["kind"] == kind]


def load_all() -> list[dict]:
    """Load every ORNL signal. Returns list of signal dicts."""
    return [load_signal(p) for p in sorted(glob.glob(os.path.join(ORNL_DIR, "*.csv")))]
