"""
continuous_monitor.py — one fridge, running continuously, watched live by the agent.

WHY (the user's ask): the benchmark and agent_eval score INDEPENDENT windows. A real
dilution fridge is ONE unbroken stream: it runs for hours at base temperature, a fault
begins at some hidden moment, and the value of an operator (human or agent) is catching
it EARLY — the lead time between fault onset and detection is the whole game.

This module simulates a single continuous run (default 24 h at 1-min cadence) from the
realistic engine, then slides a rolling window across it. At each poll the live Sentinel
agent (real Opus) looks only at the telemetry SO FAR and says whether an anomaly is
developing. We log every poll and measure:

  * detection latency : minutes from true fault onset to first agent alarm
  * false alarms      : polls that cried wolf BEFORE onset
  * the rolling MXC / 4K trace with the onset line and the alarm line, for the figure

Nothing re-implemented: the stream is cryo_engine.simulate (realistic=True), the watcher
is multi_agent._ask with the same Sentinel prompt as the panel.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import cryo_engine as CE
from . import multi_agent as MA
from . import agent as A
from . import virtual_clone as VC

DEFAULT_FINGERPRINT_DIR = "data/real/bluefors_cryometrics_sample"

SENTINEL_SYS = (
    "You are Sentinel, watching a dilution fridge run continuously. You see telemetry "
    "from the start of the run up to NOW. Stages: temp1=50K, temp2=4K(magnet flange), "
    "temp3=still, temp4=cold plate, temp5=mixing chamber(~10mK), flow, pressures p1..p6. "
    "A fault may or may not have begun. Reply ONLY JSON: "
    '{"anomaly_developing": bool, "which_channels": [str], "confidence": "low|med|high"}.')


@dataclass
class MonitorConfig:
    fault_class: str = "helium_leak"
    severity: float = 0.6
    onset_frac: float = 0.5         # fault begins halfway through the run
    hours: float = 24.0
    dt_min: float = 1.0             # 1-min telemetry cadence (real BlueFors logs at ~1 min)
    poll_every_min: float = 30.0    # agent looks every 30 min of fridge time
    window_h: float = 4.0           # rolling look-back window the agent sees
    seed: int = 7
    backend: str = "litellm"
    fingerprint_dir: str | None = DEFAULT_FINGERPRINT_DIR
    log_path: str = "outputs/continuous_monitor_turns.jsonl"
    out_path: str = "outputs/continuous_monitor.json"


def _slice(cols: dict, i0: int, i1: int) -> dict:
    return {k: (np.asarray(v)[i0:i1] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(cols["t_s"])
                else v) for k, v in cols.items()}


def run(cfg: MonitorConfig) -> dict:
    """Simulate one continuous run and watch it live with the Sentinel agent."""
    os.makedirs(os.path.dirname(cfg.log_path) or ".", exist_ok=True)
    fp = None
    if cfg.fingerprint_dir:
        try:
            fp = VC.learn_fingerprint(cfg.fingerprint_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[monitor] fingerprint unavailable ({exc})")
    engine_cfg = CE.EngineConfig(fingerprint=fp, realistic=True, imperfections=True)

    # one continuous stream
    scen = CE.Scenario(cfg.fault_class, cfg.severity, cfg.onset_frac)
    stream = CE.simulate(scen, engine_cfg, hours=cfg.hours, dt_min=cfg.dt_min, seed=cfg.seed)
    t_s = stream["t_s"]
    onset_min = cfg.onset_frac * t_s[-1] / 60.0
    win = int(round(cfg.window_h * 60 / cfg.dt_min))
    step = int(round(cfg.poll_every_min / cfg.dt_min))

    print(f"[monitor] continuous {cfg.hours:.0f} h run of '{cfg.fault_class}' "
          f"(sev {cfg.severity}), onset at t={onset_min:.0f} min. "
          f"Agent polls every {cfg.poll_every_min:.0f} min over a {cfg.window_h:.0f} h look-back.")

    logf = open(cfg.log_path, "w")
    polls = []
    first_alarm_min = None
    false_alarms = 0
    for i1 in range(win, len(t_s) + 1, step):
        i0 = max(0, i1 - win)
        now_min = t_s[i1 - 1] / 60.0
        window = _slice(stream, i0, i1)
        summary = A.summarize_window(window)
        reply = MA._ask(SENTINEL_SYS,
                        f"Telemetry from t={t_s[i0]/60.0:.0f} min to t={now_min:.0f} min:\n"
                        f"{json.dumps(summary, indent=2)}\nIs an anomaly developing?",
                        cfg.backend)
        alarm = bool(reply.get("anomaly_developing", False))
        past_onset = bool(now_min >= onset_min)
        if alarm and not past_onset:
            false_alarms += 1
        if alarm and past_onset and first_alarm_min is None:
            first_alarm_min = now_min
        rec = {"now_min": round(float(now_min), 1), "past_onset": past_onset, "alarm": alarm,
               "confidence": reply.get("confidence"), "channels": reply.get("which_channels"),
               "mxc_mK": float(np.nanmedian(window["temp5_T"][-5:]) * 1e3),
               "t4_K": float(np.nanmedian(window["temp2_T"][-5:]))}
        polls.append(rec)
        logf.write(json.dumps({**rec, "reply": reply}) + "\n"); logf.flush()
        flag = "ALARM" if alarm else "  -  "
        mark = "(post-onset)" if past_onset else "(pre-onset)"
        print(f"  t={now_min:5.0f}min {mark:12s} MXC={rec['mxc_mK']:5.1f}mK "
              f"4K={rec['t4_K']:5.2f}K  Sentinel: {flag} {reply.get('confidence','')}")
    logf.close()

    lead = None if first_alarm_min is None else round(first_alarm_min - onset_min, 1)
    result = {
        "config": asdict(cfg),
        "onset_min": round(onset_min, 1),
        "first_alarm_after_onset_min": first_alarm_min,
        "detection_latency_min": lead,
        "false_alarms_before_onset": false_alarms,
        "n_polls": len(polls),
        "polls": polls,
        "trace": {  # downsampled full stream for the figure
            "t_min": (t_s / 60.0).tolist(),
            "mxc_mK": (np.nan_to_num(stream["temp5_T"]) * 1e3).tolist(),
            "t4_K": np.nan_to_num(stream["temp2_T"]).tolist(),
        },
    }
    with open(cfg.out_path, "w") as f:
        json.dump(result, f)
    print(f"[monitor] onset t={onset_min:.0f}min, first true alarm t={first_alarm_min}min, "
          f"lead time={lead}min, false alarms={false_alarms}")
    return result
