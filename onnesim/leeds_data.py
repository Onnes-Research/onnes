"""
leeds_data.py — loader for REAL dilution-fridge telemetry.

Source (VERIFIED, downloaded, inspected): University of Leeds Research Data
Repository, dataset for "Directed delivery of terahertz frequency radiation within
a dry 3He dilution refrigerator", CC BY 4.0.
  https://archive.researchdata.leeds.ac.uk/1034/  -> Cryostat_Temperature_logs.zip
A dry DR-200 dilution refrigerator; 15 real cooldown logs (Sep 2020 - Mar 2022) in
Bluefors log format, plus Min_Temps.txt (per-cooldown base temps + cooldown time).

This is the FIRST real fridge data in the project. It lets validate_cooldown.py
compare OnnesSim/Cryowala against REAL hardware behavior instead of only a model.

Files live in data/real/leeds/ (git-ignored; download via docs/DATA_LINKS.md).
Logs are latin-1 encoded with a Windows-path first line; columns include
Time(secs), pressures (P1..P6), and stage temperatures.
"""
from __future__ import annotations
import csv
import os
import numpy as np

LEEDS_DIR = os.path.join("data", "real", "leeds", "Cryostat Temperature logs")

# Min_Temps.txt columns (from its header line, verified):
MIN_TEMPS_COLS = ["Date", "Comment", "PT1_K", "PT2_K", "Still_K", "mK100_K",
                  "MC_K", "Sample_K", "CooldownTime_s", "MagnetBottom_K"]


def load_min_temps(path: str | None = None) -> list[dict]:
    """Load Min_Temps.txt: one row per real cooldown with base temps + cooldown time."""
    path = path or os.path.join(LEEDS_DIR, "Min_Temps.txt")
    out = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.split()
            if len(vals) < len(MIN_TEMPS_COLS):
                continue
            out.append({c: float(v) for c, v in zip(MIN_TEMPS_COLS, vals)})
    return out


def _clean_header(cols: list[str]) -> list[str]:
    # Leeds logs are null-padded (UTF-16 remnants); strip NULs and whitespace.
    return [c.replace("\x00", "").strip() for c in cols]


def _clean_cell(x: str) -> str:
    return x.replace("\x00", "").strip()


def load_log(path: str, max_rows: int | None = None) -> dict:
    """Load one real Bluefors log CSV into {column_name: np.ndarray}.

    Handles the latin-1 encoding and the Windows-path first line. Columns include
    Time(secs), pressures, and temperature channels (names vary by log)."""
    with open(path, encoding="latin-1", errors="replace") as fh:
        reader = csv.reader(fh)
        first = next(reader)
        # first line may be a Windows path, not the header; detect the header row
        if len(first) == 1 or "Time" not in ",".join(first):
            header = _clean_header(next(reader))
        else:
            header = _clean_header(first)
        # Read all data rows; the header may be null-padded to a different width than
        # the data rows (verified: some logs have 193 header fields, 65 data fields).
        # Use the modal data-row width as truth and align the header to it.
        all_rows = []
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            all_rows.append(row)
    if not all_rows:
        return {}
    from collections import Counter
    width = Counter(len(r) for r in all_rows).most_common(1)[0][0]
    rows = [r for r in all_rows if len(r) == width]
    # drop empty null-only header names, then align to data width
    header = [h for h in header if h][:width]
    while len(header) < width:
        header.append(f"col{len(header)}")

    cols: dict = {}
    if not rows:
        return {name: np.array([]) for name in header}
    # build column-by-column so a bad cell in one column doesn't sink the rest
    for j, name in enumerate(header):
        raw = [_clean_cell(r[j]) for r in rows]
        try:
            cols[name] = np.array(raw, dtype=float)
        except ValueError:
            cols[name] = np.array(raw, dtype=object)
    return cols


def list_logs() -> list[str]:
    """Return paths to all real Leeds cooldown logs, sorted by time."""
    if not os.path.isdir(LEEDS_DIR):
        return []
    return sorted(os.path.join(LEEDS_DIR, f) for f in os.listdir(LEEDS_DIR)
                  if f.startswith("log ") and f.endswith(".csv"))


def find_temp_columns(cols: dict) -> dict:
    """Map our stage names to this log's TEMPERATURE columns (the 'T(K)' ones, not
    the 't(s)' timestamp or 'R(Ohm)' resistance columns). Matched on real names:
    PT1 Head/Plate, PT2 Head/Plate, Still, 100mK Plate, MC Cernox."""
    def temp_col(*subs):
        for name in cols:
            low = name.lower()
            if "t(k)" in low and all(s in low for s in subs):
                return name
        return None
    mapping = {}
    for stage, subs in [("50K", ("pt1",)), ("4K", ("pt2",)), ("Still", ("still",)),
                        ("CP", ("100mk",)), ("MXC", ("mc",))]:
        c = temp_col(*subs)
        if c:
            mapping[stage] = c
    return mapping
