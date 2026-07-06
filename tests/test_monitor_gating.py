"""Tests for monitor_gating — the confidence-gate trade-off on the continuous monitor.
Uses a tiny synthetic poll set so it never depends on a live run."""
import json

from onnesim import monitor_gating as MG


def _fake_monitor(tmp_path):
    polls = [
        # pre-onset: 2 low false alarms, 1 high false alarm
        {"now_min": 100, "past_onset": False, "alarm": True, "confidence": "low"},
        {"now_min": 130, "past_onset": False, "alarm": True, "confidence": "low"},
        {"now_min": 160, "past_onset": False, "alarm": True, "confidence": "high"},
        {"now_min": 190, "past_onset": False, "alarm": False, "confidence": "low"},
        # post-onset: first true alarm is med, later ones high
        {"now_min": 220, "past_onset": True, "alarm": True, "confidence": "med"},
        {"now_min": 250, "past_onset": True, "alarm": True, "confidence": "high"},
    ]
    d = {"onset_min": 200.0, "config": {"poll_every_min": 30.0}, "polls": polls}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_gating_reduces_false_alarms(tmp_path):
    src = _fake_monitor(tmp_path)
    out = MG.analyze(src, out_path=str(tmp_path / "out.json"))
    gates = {g["gate"]: g for g in out["gates"]}
    # any: 3 false alarms; high: only the 1 high-confidence false alarm survives
    assert gates["any"]["false_alarms_before_onset"] == 3
    assert gates["high"]["false_alarms_before_onset"] == 1
    # med gate drops the two low false alarms -> 1 (the high one)
    assert gates["med"]["false_alarms_before_onset"] == 1


def test_gating_latency_tradeoff(tmp_path):
    src = _fake_monitor(tmp_path)
    out = MG.analyze(src, out_path=str(tmp_path / "out.json"))
    gates = {g["gate"]: g for g in out["gates"]}
    # any/med detect at the med alarm (t=220 -> latency 20)
    assert gates["any"]["detection_latency_min"] == 20.0
    # high must wait for the high alarm (t=250 -> latency 50): precision costs latency
    assert gates["high"]["detection_latency_min"] == 50.0
    assert all(g["detected"] for g in out["gates"])
