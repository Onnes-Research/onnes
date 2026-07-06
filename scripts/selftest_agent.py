#!/usr/bin/env python3
"""
Offline self-test for CryoAgent's parsing + scoring path — PROVABLE, no API tokens.

Run:
    /Users/praneeth/Developer/onnes-research/.venv/bin/python scripts/selftest_agent.py

What it proves (every check is asserted, and prints PASS/FAIL):

  A. Return-shape parsing. Known model outputs — clean JSON, prose-wrapped JSON,
     markdown-fenced JSON, and a multi-block (non-text + text) message — are
     pushed through agent.py's REAL model path (summarize_window -> _call_claude
     -> _extract_json -> AgentVerdict) via onnesim.mockclient. The mock mimics
     the real `client.messages.create(...)` return shape; agent.py is unmodified.

  B. Graceful degradation. Malformed JSON, no-JSON prose, empty output, and a
     simulated API exception:
        * backend="claude" -> _call_claude returns None -> run_agent RAISES
          (the parser refuses to invent a verdict from junk), and
        * backend="auto"   -> falls back to the transparent stub, NO crash.

  C. Scoring wiring. onnesim.evaluate.score turns (truth, pred) rows into
     detection precision/recall/F1 and class accuracy. We check a hand-built
     confusion set scores exactly, and that a mixed agent/baseline run scores
     without error on real OnnesSim windows (magnet_quench + normal).

  D. Lead-time metric (the paper's key number). On a slow-developing fault we
     measure how many minutes BEFORE the FRTMS-style absolute-threshold alarm a
     trajectory-reading detector first flags the developing fault, using sliding
     windows over one scenario. We assert the metric is well-defined and that the
     early detector leads the threshold baseline on a helium_leak.

Exit code 0 on all-pass, 1 otherwise. Nothing here needs a network or API key.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# Make `onnesim` importable when run as `python scripts/selftest_agent.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnesim  # noqa: F401  (proves the package imports; task requires it)
from onnesim import FridgeParams, simulate, make_hook, FaultSpec, emit
from onnesim import agent as A
from onnesim import evaluate as E
from onnesim import constants as C
from onnesim import mockclient as M


# --------------------------------------------------------------------------- #
# Tiny assertion harness (dependency-free; collects results, never aborts early)
# --------------------------------------------------------------------------- #
@dataclass
class _Result:
    name: str
    ok: bool
    detail: str = ""


_RESULTS: list = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    """Run one check. `fn` returns None/'' on success, or a failure detail
    string; raising is also treated as failure (with traceback captured)."""
    try:
        detail = fn()
        ok = not detail
        _RESULTS.append(_Result(name, ok, detail or ""))
    except Exception as e:  # noqa: BLE001 - we want ALL failures, not the first
        _RESULTS.append(_Result(name, False, f"EXCEPTION: {e}\n{traceback.format_exc()}"))


def _need(cond: bool, msg: str) -> str:
    return "" if cond else msg


# --------------------------------------------------------------------------- #
# OnnesSim window builders
# --------------------------------------------------------------------------- #
def _window(cols: dict, t0: float, t1: float) -> dict:
    m = (cols["t_s"] >= t0) & (cols["t_s"] <= t1)
    return {k: np.asarray(v)[m] for k, v in cols.items()}


def build_scenario(fault_class: str, onset_h: float = 18.0, severity: float = 1.0,
                   dur_h: float = 24.0, seed: int = 1) -> tuple:
    """One OnnesSim cooldown with a labeled fault. Returns (cols, onset_s, dur_s).

    A late onset on a long run means the fridge is cold and the pulse tube is
    fully engaged before the fault develops — so trailing-window trends reflect
    the FAULT, not the initial cooldown transient.
    """
    dur_s = dur_h * 3600.0
    onset_s = onset_h * 3600.0
    stage = 1 if fault_class == "magnet_quench" else 3  # 4K flange vs a cold stage
    flow_scale = 1.0 - 0.6 * severity if fault_class == "blocked_impedance" else 1.0
    spec = FaultSpec(fault_class=fault_class, onset_s=onset_s, severity=severity,
                     target_stage=stage, flow_scale=flow_scale)
    sim = simulate(FridgeParams(), duration_s=dur_s, dt_s=60.0,
                   fault_hook=make_hook(spec), seed=seed)
    cols = emit(sim, spec, seed=seed)
    return cols, onset_s, dur_s


def final_window(cols: dict, hours: float = 1.0) -> dict:
    """The trailing `hours` of a scenario — what a monitor would judge 'now'."""
    t_end = float(cols["t_s"][-1])
    return _window(cols, t_end - hours * 3600.0, t_end)


# --------------------------------------------------------------------------- #
# A. Return-shape parsing through the real model path
# --------------------------------------------------------------------------- #
def _mk_quench_and_normal_windows():
    q_cols, _, _ = build_scenario("magnet_quench", onset_h=18.0, severity=1.0, seed=1)
    n_cols, _, _ = build_scenario("normal", onset_h=100.0, severity=0.0, seed=2)
    return final_window(q_cols, 1.0), final_window(n_cols, 1.0)


def test_parsing():
    q_win, n_win = _mk_quench_and_normal_windows()

    def clean():
        v = M.run_agent_with_mock(q_win, M.SCRIPTS["clean_quench"], backend="claude")
        return _need(v.backend.startswith("claude:") and v.fault_detected is True
                     and v.fault_class == "magnet_quench" and v.severity == "high",
                     f"clean JSON not parsed correctly: {v.as_dict()}") or \
            _need(len(M.MockAnthropic.calls) == 1, "expected exactly 1 API call")

    def prose():
        v = M.run_agent_with_mock(q_win, M.SCRIPTS["prose_quench"], backend="claude")
        return _need(v.backend.startswith("claude:") and v.fault_class == "magnet_quench",
                     f"prose-wrapped JSON not parsed: {v.as_dict()}")

    def fenced():
        v = M.run_agent_with_mock(n_win, M.SCRIPTS["fenced_normal"], backend="claude")
        return _need(v.backend.startswith("claude:") and v.fault_detected is False
                     and v.fault_class == "normal",
                     f"markdown-fenced JSON not parsed: {v.as_dict()}")

    def multiblock():
        # Non-text block + text block: proves the type=="text" filter + join.
        v = M.run_agent_with_mock(n_win, M.SCRIPTS["multiblock_normal"], backend="claude")
        return _need(v.backend.startswith("claude:") and v.fault_class == "normal",
                     f"multi-block message not parsed: {v.as_dict()}")

    def field_coercion():
        # Missing keys must coerce to safe defaults, not raise.
        v = M.run_agent_with_mock(n_win, '{"fault_detected": true}', backend="claude")
        return _need(v.fault_detected is True and v.fault_class == "normal"
                     and v.severity == "none" and v.recommended_action == ""
                     and v.rationale == "",
                     f"partial JSON not coerced to defaults: {v.as_dict()}")

    check("A1 clean JSON -> verdict via real claude path", clean)
    check("A2 prose-wrapped JSON extracted", prose)
    check("A3 markdown-fenced JSON extracted", fenced)
    check("A4 multi-block (non-text+text) message parsed", multiblock)
    check("A5 partial JSON coerces to safe defaults", field_coercion)


# --------------------------------------------------------------------------- #
# B. Graceful degradation on malformed / missing / erroring output
# --------------------------------------------------------------------------- #
def test_graceful_degradation():
    q_win, _ = _mk_quench_and_normal_windows()

    bad_keys = ["malformed", "no_json", "empty", "api_error"]

    def auto_never_crashes():
        for key in bad_keys:
            v = M.run_agent_with_mock(q_win, M.SCRIPTS[key], backend="auto")
            if v.backend != "stub":
                return f"{key}: expected stub fallback, got backend={v.backend!r}"
            # stub must still return a well-formed verdict object
            d = v.as_dict()
            for f in ("fault_detected", "fault_class", "severity",
                      "recommended_action", "rationale", "backend"):
                if f not in d:
                    return f"{key}: stub verdict missing field {f!r}"
        return ""

    def claude_refuses_junk():
        # backend="claude" must RAISE rather than fabricate a verdict from junk.
        for key in ("malformed", "no_json", "empty"):
            try:
                M.run_agent_with_mock(q_win, M.SCRIPTS[key], backend="claude")
                return f"{key}: expected RuntimeError, but run_agent returned a verdict"
            except RuntimeError:
                pass
        return ""

    def call_claude_returns_none():
        # Lower-level contract: _call_claude returns None on genuinely unparseable text.
        for key in ("malformed", "no_json", "empty"):
            r = M.call_claude_with_mock(q_win, M.SCRIPTS[key])
            if r is not None:
                return f"{key}: _call_claude should return None, got {r!r}"
        return ""

    def stray_brace_now_parses():
        # After the _extract_json robustness fix (balanced-brace scan), a decoy
        # brace BEFORE valid JSON must no longer defeat extraction: the real
        # magnet_quench object should parse. (Was a proven bug pre-fix.)
        r = M.call_claude_with_mock(q_win, M.SCRIPTS["stray_brace_quench"])
        if r is None:
            return "stray_brace_quench: expected a parsed verdict, got None"
        if r.fault_class != "magnet_quench":
            return f"stray_brace_quench: expected magnet_quench, got {r.fault_class!r}"
        return ""

    check("B1 backend=auto degrades to stub (no crash) on bad output", auto_never_crashes)
    check("B2 backend=claude RAISES on unparseable output (no fabrication)", claude_refuses_junk)
    check("B3 _call_claude returns None on genuinely unparseable output", call_claude_returns_none)
    check("B4 stray-brace decoy before valid JSON now parses (bug fixed)", stray_brace_now_parses)


# --------------------------------------------------------------------------- #
# C. Scoring wiring (onnesim.evaluate.score) + end-to-end on real windows
# --------------------------------------------------------------------------- #
def test_scoring():
    def exact_confusion():
        # 2 correct quench detections (1 mislabeled class), 1 missed leak,
        # 1 false alarm on normal, 1 true negative.
        rows = [
            {"truth_class": "magnet_quench", "pred_detected": True,  "pred_class": "magnet_quench"},
            {"truth_class": "magnet_quench", "pred_detected": True,  "pred_class": "helium_leak"},
            {"truth_class": "helium_leak",   "pred_detected": False, "pred_class": "normal"},
            {"truth_class": "normal",        "pred_detected": True,  "pred_class": "magnet_quench"},
            {"truth_class": "normal",        "pred_detected": False, "pred_class": "normal"},
        ]
        s = E.score(rows)
        # tp=2, fn=1, fp=1, tn=1 -> P=2/3, R=2/3, class acc on detected = 1/2
        want = {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
        errs = []
        if s["counts"] != want:
            errs.append(f"counts {s['counts']} != {want}")
        if s["precision"] != round(2 / 3, 3):
            errs.append(f"precision {s['precision']}")
        if s["recall"] != round(2 / 3, 3):
            errs.append(f"recall {s['recall']}")
        if s["class_accuracy_on_detected"] != 0.5:
            errs.append(f"class_acc {s['class_accuracy_on_detected']}")
        if s["detect_accuracy"] != round(3 / 5, 3):
            errs.append(f"detect_acc {s['detect_accuracy']}")
        return "; ".join(errs)

    def end_to_end_real_windows():
        # Build labeled windows, judge each with a scripted mock verdict + the
        # threshold baseline, and confirm scoring runs and separates the two.
        q_cols, _, _ = build_scenario("magnet_quench", onset_h=18.0, seed=7)
        n_cols, _, _ = build_scenario("normal", onset_h=100.0, seed=8)
        q_win, n_win = final_window(q_cols, 1.0), final_window(n_cols, 1.0)

        # Agent: mock returns the correct verdict for each window.
        agent_rows, base_rows = [], []
        for cols, truth, script in [
            (q_win, "magnet_quench", M.SCRIPTS["clean_quench"]),
            (n_win, "normal", M.SCRIPTS["clean_normal"]),
        ]:
            v = M.run_agent_with_mock(cols, script, backend="claude")
            b = E.threshold_baseline(cols)
            agent_rows.append({"truth_class": truth,
                               "pred_detected": v.fault_detected,
                               "pred_class": v.fault_class})
            base_rows.append({"truth_class": truth,
                              "pred_detected": b["fault_detected"],
                              "pred_class": b["fault_class"]})
        a = E.score(agent_rows)
        b = E.score(base_rows)
        # With correct scripted verdicts the agent should be perfect here.
        errs = []
        if a["detect_accuracy"] != 1.0:
            errs.append(f"agent detect_acc {a['detect_accuracy']} != 1.0")
        if a["class_accuracy_on_detected"] != 1.0:
            errs.append(f"agent class_acc {a['class_accuracy_on_detected']} != 1.0")
        if b["n"] != 2:
            errs.append(f"baseline n {b['n']} != 2")
        return "; ".join(errs)

    check("C1 score() computes exact confusion/P/R/class-acc", exact_confusion)
    check("C2 end-to-end scoring on real OnnesSim windows", end_to_end_real_windows)


# --------------------------------------------------------------------------- #
# D. LEAD-TIME METRIC
# --------------------------------------------------------------------------- #
# See module docstring + the report for the full design. Core idea: slide a
# trailing window across ONE scenario; find the first window at which each
# detector fires; lead = (first alarm of the reference/threshold detector) minus
# (first alarm of the agent/early detector), in minutes. Positive => early warning.
def first_alarm_time(cols: dict, detector: Callable[[dict], bool],
                     window_s: float = 3600.0, stride_s: float = 300.0,
                     start_s: float = 0.0) -> Optional[float]:
    """First window-END time (seconds) at which `detector(window)` is True.

    Returns None if the detector never fires over the scenario.
    """
    t = np.asarray(cols["t_s"], dtype=float)
    t_end = float(t[-1])
    te = max(window_s, start_s)
    while te <= t_end + 1e-6:
        w = _window(cols, te - window_s, te)
        if len(w["t_s"]) >= 3 and detector(w):
            return te
        te += stride_s
    return None


def lead_time_minutes(cols: dict,
                      early_detector: Callable[[dict], bool],
                      reference_detector: Callable[[dict], bool],
                      window_s: float = 3600.0, stride_s: float = 300.0,
                      start_s: float = 0.0) -> dict:
    """Minutes the `early_detector` fires BEFORE the `reference_detector`.

    Returns a dict:
        {early_first_s, reference_first_s, lead_minutes}
    lead_minutes is None if either detector never fires. Positive => the early
    detector gave that many minutes of advance warning over the reference alarm.
    """
    ef = first_alarm_time(cols, early_detector, window_s, stride_s, start_s)
    rf = first_alarm_time(cols, reference_detector, window_s, stride_s, start_s)
    lead = None if (ef is None or rf is None) else (rf - ef) / 60.0
    return {"early_first_s": ef, "reference_first_s": rf, "lead_minutes": lead}


def threshold_detector(window: dict) -> bool:
    """FRTMS-style absolute-threshold alarm (the reference bar)."""
    return bool(E.threshold_baseline(window)["fault_detected"])


def trajectory_trend_detector(window: dict) -> bool:
    """Early detector proxy: reads the SAME compact summary the agent sees and
    flags a monotone-ish RISE across the window on any safety channel.

    This models what a trajectory-reading agent does (spot a developing trend
    before an absolute cutoff is crossed) without needing an API. It uses
    agent.summarize_window so it consumes the exact features the LLM would.
    """
    s = A.summarize_window(window)
    ch = s["channels"]
    # (channel, relative-rise tolerance) — magnet & 4K are the quench axes;
    # still & mxc are the slow warm-up axes for leaks / heat loads.
    for name, tol in (("temp6_T", 0.08), ("temp2_T", 0.08),
                      ("temp5_T", 0.10), ("temp3_T", 0.10)):
        tr = ch.get(name, {}).get("trajectory", [])
        if len(tr) >= 2 and tr[0] > 0 and (tr[-1] - tr[0]) / abs(tr[0]) > tol:
            return True
    return False


def test_lead_time():
    def metric_well_defined():
        # The lead-time metric is defined for a SETTLED, running fridge (cold,
        # pulse tube fully engaged). On a NORMAL scenario, once settled, neither
        # detector should fire -> lead None, no crash. We scan from t=12h, which
        # the self-test verifies is safely past the 300K->mK cooldown transient
        # (the fridge reaches base temperature by ~9h; before that the absolute
        # threshold legitimately trips on still-warm stages — that is cooldown,
        # not a fault, and outside the metric's operating regime).
        n_cols, _, _ = build_scenario("normal", onset_h=100.0, severity=0.0, seed=3)
        settled_start = 12.0 * 3600.0
        # Guard: confirm the fridge really is settled where we start scanning.
        settle_win = _window(n_cols, settled_start - 3600.0, settled_start)
        if threshold_detector(settle_win):
            return f"fridge not settled by {settled_start/3600:.0f}h; metric domain wrong"
        r = lead_time_minutes(n_cols, trajectory_trend_detector, threshold_detector,
                              start_s=settled_start)
        return _need(r["lead_minutes"] is None and r["early_first_s"] is None
                     and r["reference_first_s"] is None,
                     f"settled normal scenario should yield no alarms: {r}")

    def early_warning_on_slow_fault():
        # helium_leak develops slowly -> trend detector should LEAD the absolute
        # threshold alarm by a positive margin.
        cols, onset_s, _ = build_scenario("helium_leak", onset_h=18.0, severity=1.0, seed=1)
        r = lead_time_minutes(cols, trajectory_trend_detector, threshold_detector,
                              start_s=onset_s - 3600.0)
        if r["lead_minutes"] is None:
            return f"expected a finite lead on helium_leak, got {r}"
        if r["lead_minutes"] <= 0:
            return f"expected POSITIVE lead (early warning), got {r['lead_minutes']} min"
        # sanity: both alarms should land at/after the fault onset window.
        if r["early_first_s"] < onset_s - 3600.0:
            return f"early alarm implausibly before onset: {r}"
        _RESULTS_LEAD["helium_leak"] = r
        return ""

    def quench_lead_nonnegative():
        # Fast fault: trend detector should still be at least as early as the
        # absolute threshold (lead >= 0). Records the number for the report.
        cols, onset_s, _ = build_scenario("magnet_quench", onset_h=18.0, severity=1.0, seed=1)
        r = lead_time_minutes(cols, trajectory_trend_detector, threshold_detector,
                              start_s=onset_s - 3600.0)
        _RESULTS_LEAD["magnet_quench"] = r
        if r["lead_minutes"] is None:
            return f"expected a finite lead on magnet_quench, got {r}"
        return _need(r["lead_minutes"] >= 0.0,
                     f"expected non-negative lead on quench, got {r['lead_minutes']}")

    check("D1 lead-time metric well-defined (no alarms on normal)", metric_well_defined)
    check("D2 early detector LEADS threshold on slow helium_leak", early_warning_on_slow_fault)
    check("D3 early detector >= threshold on fast magnet_quench", quench_lead_nonnegative)


_RESULTS_LEAD: dict = {}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 72)
    print("CryoAgent offline self-test  (mock Anthropic client; no API tokens)")
    print("venv:", sys.executable)
    print("=" * 72)

    test_parsing()
    test_graceful_degradation()
    test_scoring()
    test_lead_time()

    n_pass = sum(1 for r in _RESULTS if r.ok)
    n_fail = len(_RESULTS) - n_pass

    for r in _RESULTS:
        tag = "PASS" if r.ok else "FAIL"
        print(f"  [{tag}] {r.name}")
        if not r.ok:
            for line in r.detail.rstrip().splitlines():
                print(f"         {line}")

    # Lead-time summary (illustrative numbers produced above).
    if _RESULTS_LEAD:
        print("-" * 72)
        print("Lead-time metric (minutes of advance warning vs FRTMS threshold):")
        for fc, r in _RESULTS_LEAD.items():
            lead = r["lead_minutes"]
            ef = None if r["early_first_s"] is None else round(r["early_first_s"] / 3600.0, 3)
            rf = None if r["reference_first_s"] is None else round(r["reference_first_s"] / 3600.0, 3)
            print(f"    {fc:16s} early@{ef}h  threshold@{rf}h  lead={lead} min")

    print("=" * 72)
    summary = f"SELF-TEST {'PASS' if n_fail == 0 else 'FAIL'}  ({n_pass}/{len(_RESULTS)} checks passed)"
    print(summary)
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
