"""
monitor_sweep.py — the multi-seed / multi-fault continuous-monitoring study.

WHY THIS EXISTS
---------------
The paper's continuous-monitoring section (Sec. 8) is a single seed, single fault: one
24 h helium-leak run in which the Sentinel alarms 29.5 min after onset. Every reviewer
flagged the n=1 as the weakest part. This module runs the SAME live-Sentinel monitor
(continuous_monitor.run, real Opus polls) across a grid of faults x seeds, then aggregates
into the statistics a monitoring claim actually needs:

  * detection-latency DISTRIBUTION (median + spread) per fault and overall, not one number;
  * a precision / recall / false-alarm trade-off as the confidence GATE tightens
    (any -> med -> high), computed by re-scoring the logged polls — the paper already
    shows this for one run; here it is aggregated across the whole grid;
  * a detection-rate (did the agent ever alarm post-onset?) per fault.

Nothing is re-implemented: each grid cell is one continuous_monitor.run(); the gate
re-scoring reuses monitor_gating's confidence ranking. Every number therefore traces to
a logged poll from a real agent turn, exactly as the single-run study does.

HONEST SCOPE
------------
This is still a SIMULATED fridge (physics + fingerprint), and latency remains cadence-
bounded by the poll interval (the smallest observable latency is one poll). What the
sweep adds over the single run is variance: how latency and false-alarm rate move across
faults and noise realizations, which is what "monitoring readiness" requires. It does not
add real-hardware validation (no labeled real fault data exists; see docs).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np

from . import continuous_monitor as CM

# Confidence ranking shared with monitor_gating (kept local to avoid a hard import cycle).
_RANK = {"low": 0, "med": 1, "medium": 1, "high": 2, None: 0, "": 0}
_GATES = ("any", "med", "high")

# The faults worth watching develop over hours (slow/among-confusable thermal faults +
# the sharp quench). heat_load_spike is a fast step; included so the grid spans dynamics.
DEFAULT_FAULTS = ("helium_leak", "blocked_impedance", "wiring_heat_ingress",
                  "magnet_quench", "heat_load_spike")


def _passes(conf, gate: str) -> bool:
    need = {"any": 0, "med": 1, "high": 2}[gate]
    return _RANK.get(conf, 0) >= need


@dataclass
class SweepConfig:
    faults: tuple[str, ...] = DEFAULT_FAULTS
    seeds: tuple[int, ...] = (7, 11, 23, 42, 101)   # 5 noise realizations
    severity: float = 0.6
    onset_frac: float = 0.5
    hours: float = 24.0
    dt_min: float = 1.0
    poll_every_min: float = 30.0
    window_h: float = 4.0
    backend: str = "litellm"
    workers: int = 1                # parallel CELLS (each cell's polls stay serial within it)
    fingerprint_dir: str | None = CM.DEFAULT_FINGERPRINT_DIR
    out_dir: str = "outputs/monitor_sweep"
    out_path: str = "outputs/monitor_sweep.json"


def _gate_scores(polls: list[dict], onset_min: float, cadence: float) -> dict:
    """Per-gate (false_alarms, latency, detected, tp, fp, fn-ish) from logged polls.

    A poll fires under a gate iff it alarmed AND its confidence >= the gate threshold.
    - false alarm  : fires && pre-onset
    - true alarm   : fires && post-onset
    - detection latency : (first true-alarm time) - onset, or None if never
    Recall over the post-onset polls = fraction of post-onset polls that fire (how
    persistently the agent holds the alarm once the fault is present); precision =
    true fires / all fires. These are the trade-off curves the reviewers asked for."""
    out = {}
    for gate in _GATES:
        fires_pre = fires_post = post_total = 0
        first_after = None
        for p in polls:
            post = bool(p["past_onset"])
            post_total += int(post)
            if bool(p["alarm"]) and _passes(p.get("confidence"), gate):
                if post:
                    fires_post += 1
                    if first_after is None:
                        first_after = float(p["now_min"])
                else:
                    fires_pre += 1
        total_fires = fires_pre + fires_post
        latency = None if first_after is None else round(first_after - onset_min, 1)
        out[gate] = {
            "false_alarms": fires_pre,
            "true_alarms": fires_post,
            "detection_latency_min": latency,
            "detected": first_after is not None,
            "precision": (fires_post / total_fires) if total_fires else None,
            "recall_post_onset": (fires_post / post_total) if post_total else None,
        }
    return out


def _run_cell(fault: str, seed: int, cfg: SweepConfig) -> dict:
    """Run one (fault, seed) cell: a single continuous_monitor.run() + gate scoring.
    Independent of every other cell (unique log/out paths) -> safe to thread."""
    tag = f"{fault}_seed{seed}"
    mcfg = CM.MonitorConfig(
        fault_class=fault, severity=cfg.severity, onset_frac=cfg.onset_frac,
        hours=cfg.hours, dt_min=cfg.dt_min, poll_every_min=cfg.poll_every_min,
        window_h=cfg.window_h, seed=seed, backend=cfg.backend,
        fingerprint_dir=cfg.fingerprint_dir,
        log_path=f"{cfg.out_dir}/{tag}_turns.jsonl",
        out_path=f"{cfg.out_dir}/{tag}.json")
    res = CM.run(mcfg)
    gates = _gate_scores(res["polls"], res["onset_min"], cfg.poll_every_min)
    return {
        "fault": fault, "seed": seed,
        "onset_min": res["onset_min"], "n_polls": res["n_polls"],
        "detection_latency_min": res["detection_latency_min"],
        "false_alarms_before_onset": res["false_alarms_before_onset"],
        "gates": gates,
    }


def run(cfg: SweepConfig) -> dict:
    """Run the whole fault x seed grid live and aggregate. Each cell is one
    continuous_monitor.run() (real Sentinel polls); per-cell logs land in cfg.out_dir.

    Cells are independent, so with cfg.workers>1 they run in a thread pool (the polls
    WITHIN a cell stay serial — a continuous run is inherently sequential in time)."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    grid = [(f, s) for f in cfg.faults for s in cfg.seeds]
    cells: list[dict] = []
    if cfg.workers <= 1:
        for fault, seed in grid:
            print(f"\n===== monitor sweep cell: {fault}_seed{seed} =====")
            cells.append(_run_cell(fault, seed, cfg))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[monitor_sweep] running {len(grid)} cells on {cfg.workers} parallel workers "
              f"(polls within a cell stay serial)")
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(_run_cell, f, s, cfg): (f, s) for f, s in grid}
            done = 0
            for fut in as_completed(futs):
                f, s = futs[fut]
                cell = fut.result()
                cells.append(cell)
                done += 1
                lat = cell["detection_latency_min"]
                print(f"[monitor_sweep] cell {done}/{len(grid)} done: {f}_seed{s} "
                      f"latency={lat}min false_alarms={cell['false_alarms_before_onset']}")

    summary = aggregate(cells, cfg)
    with open(cfg.out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[monitor_sweep] wrote {cfg.out_path} "
          f"({len(cells)} cells: {len(cfg.faults)} faults x {len(cfg.seeds)} seeds)")
    return summary


def _stats(xs: list[float]) -> dict | None:
    v = [float(x) for x in xs if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {"n": len(v), "median": round(float(np.median(a)), 1),
            "mean": round(float(a.mean()), 1),
            "min": round(float(a.min()), 1), "max": round(float(a.max()), 1),
            "p25": round(float(np.percentile(a, 25)), 1),
            "p75": round(float(np.percentile(a, 75)), 1)}


def aggregate(cells: list[dict], cfg: SweepConfig) -> dict:
    """Roll the per-cell results into per-fault and overall distributions + gate curves.

    Pure function of the cells (no runs), so it can be re-called on a partial/rescued grid.
    """
    faults = sorted({c["fault"] for c in cells})
    per_fault = {}
    for fc in faults:
        cs = [c for c in cells if c["fault"] == fc]
        lat = [c["detection_latency_min"] for c in cs]
        det = [c["gates"]["any"]["detected"] for c in cs]
        per_fault[fc] = {
            "n_cells": len(cs),
            "detection_rate": round(float(np.mean(det)), 3),
            "latency_min_any_gate": _stats(lat),
            "false_alarms_any_gate": _stats([c["false_alarms_before_onset"] for c in cs]),
        }

    # overall gate trade-off: pool every cell's per-gate numbers
    gate_curve = {}
    for gate in _GATES:
        fa = [c["gates"][gate]["false_alarms"] for c in cells]
        lat = [c["gates"][gate]["detection_latency_min"] for c in cells]
        det = [c["gates"][gate]["detected"] for c in cells]
        prec = [c["gates"][gate]["precision"] for c in cells if c["gates"][gate]["precision"] is not None]
        rec = [c["gates"][gate]["recall_post_onset"] for c in cells if c["gates"][gate]["recall_post_onset"] is not None]
        gate_curve[gate] = {
            "false_alarms_total": int(np.sum(fa)),
            "false_alarms_per_run": _stats(fa),
            "detection_rate": round(float(np.mean(det)), 3),
            "latency_min": _stats(lat),
            "mean_precision": round(float(np.mean(prec)), 3) if prec else None,
            "mean_recall_post_onset": round(float(np.mean(rec)), 3) if rec else None,
        }

    all_lat = [c["detection_latency_min"] for c in cells]
    all_fa = [c["false_alarms_before_onset"] for c in cells]
    return {
        "config": asdict(cfg),
        "n_cells": len(cells),
        "faults": faults,
        "seeds": list(cfg.seeds),
        "overall": {
            "detection_latency_min": _stats(all_lat),
            "false_alarms_before_onset": _stats(all_fa),
            "detection_rate_any_gate": round(
                float(np.mean([c["gates"]["any"]["detected"] for c in cells])), 3),
        },
        "per_fault": per_fault,
        "gate_curve": gate_curve,
        "cells": cells,
        "reading": (
            "Detection latency is cadence-bounded (>= one poll of "
            f"{cfg.poll_every_min:.0f} min). The gate_curve is the aggregate of the "
            "single-run gating result across the whole grid: tightening the confidence "
            "gate trades false alarms for latency, now with cross-seed/fault spread."),
    }


def reaggregate_from_dir(out_dir: str, cfg: SweepConfig | None = None) -> dict:
    """Rebuild the summary from the per-cell JSONs on disk (sweep interruption-proof)."""
    cfg = cfg or SweepConfig(out_dir=out_dir)
    cells = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".json") or fname.endswith("_turns.jsonl"):
            continue
        d = json.load(open(os.path.join(out_dir, fname)))
        if "polls" not in d:
            continue
        gates = _gate_scores(d["polls"], d["onset_min"], cfg.poll_every_min)
        cells.append({
            "fault": d["config"]["fault_class"], "seed": d["config"]["seed"],
            "onset_min": d["onset_min"], "n_polls": d["n_polls"],
            "detection_latency_min": d["detection_latency_min"],
            "false_alarms_before_onset": d["false_alarms_before_onset"],
            "gates": gates,
        })
    return aggregate(cells, cfg)
