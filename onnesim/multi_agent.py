"""
multi_agent.py — the 5-6 agent CryoAgent stack, run LIVE on the cryo engine.

Roles (each a real Claude call via the litellm proxy, or the stub offline):
  1. Sentinel     — scans a telemetry window, flags whether something is developing.
  2. Diagnostician— if flagged, classifies the fault + severity from the physics.
  3. Operator     — proposes ONE corrective action.
  4. Guardian     — safety check: approves/blocks the action against invariants.
  5. Twin         — rehearses: would the action help vs doing nothing? (uses the engine)
  6. Supervisor   — final call, reconciles the panel into a verdict.

This is the "multiple agents, multiple turns" test: a real multi-agent loop reasoning
over one telemetry window from the unified real-physics+real-noise engine.

TWO EVIDENCE-BACKED LEVERS (optional; defaults preserve exact zero-shot behavior):
  * few_shot_block: labeled contrastive telemetry examples injected into the
    Diagnostician's prompt. Curated few-shot (not many-shot) is the lever for the
    confusable-pair classification weakness; many-shot ICL is shown to break down on
    reasoning tasks (arXiv:2605.13511, 2026), and "just prompt harder" gives ~no gain
    (arXiv:2507.15066, Time-RA 2026) — it's the EXAMPLES that move the needle.
  * sc_samples: self-consistency — sample the Diagnostician N times and majority-vote
    the fault_class (Wang et al. 2203.11171; +2.2% and best cost/accuracy point vs
    self-refine which HURTS -4.6..-9.1%, arXiv:2604.22273 2026; arXiv:2502.02533 2025).
    We deliberately AVOID multi-round debate/self-refine: their apparent gains are
    largely a test-time-compute artifact (arXiv:2606.13003 "Illusion of Multi-Agent
    Advantage", 2026).
"""
from __future__ import annotations
import json
from collections import Counter

import numpy as np

from . import agent as A
from . import cryo_engine as CE


def _gemini_client():
    """Lazily build and cache a Google GenAI client (SDK), keyed by nothing (one per
    process). Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment."""
    global _GEMINI_CLIENT
    try:
        return _GEMINI_CLIENT
    except NameError:
        pass
    import os
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set for backend='gemini'")
    _GEMINI_CLIENT = genai.Client(api_key=key)
    return _GEMINI_CLIENT


def _ask_gemini(role_system: str, user: str, temperature: float | None) -> dict:
    """One agent turn via the Google GenAI SDK (gemini-3.1-pro-preview by default).

    Mirrors the litellm path exactly except for transport: same system+user split,
    same A._extract_json parser, same {'_error'/'_raw'} fallbacks — so the ONLY
    variable vs. the Claude backend is the model, keeping the comparison fair.
    Retries transient/rate-limit errors with capped exponential backoff."""
    import os, time
    from google.genai import types
    model = os.environ.get("ONNES_GEMINI_MODEL", "gemini-3.1-pro-preview")
    client = _gemini_client()
    # Hard per-request timeout so a hung socket cannot park a worker thread forever (proven
    # failure mode: a long parallel run stalled at 0% CPU because one dropped connection had
    # no client-side timeout). http_options.timeout is in MILLISECONDS in the GenAI SDK.
    timeout_ms = int(float(os.environ.get("ONNES_GEMINI_TIMEOUT_S", "90")) * 1000)
    cfg = types.GenerateContentConfig(
        system_instruction=role_system,
        max_output_tokens=8192,  # reasoning models spend tokens before the final JSON
        temperature=(float(temperature) if temperature is not None else None),
        thinking_config=types.ThinkingConfig(thinking_budget=512),
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    last = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(model=model, contents=user, config=cfg)
            text = resp.text or ""
            if not text:
                last = "empty response"
                continue
            return A._extract_json(text) or {"_raw": text[:200]}
        except Exception as e:  # noqa: BLE001
            last = str(e)
            msg = last.lower()
            transient = any(s in msg for s in ("429", "resource_exhausted", "rate",
                            "503", "unavailable", "500", "internal", "timeout", "deadline"))
            if attempt < 4 and transient:
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
                continue
            break
    return {"_error": last or "gemini call failed"}


def _ask(role_system: str, user: str, backend: str, temperature: float | None = None) -> dict:
    """One agent turn. Returns parsed JSON dict (or a safe default).

    temperature is only sent when explicitly given (self-consistency sampling needs
    diversity); leaving it None keeps the original deterministic-ish default so the
    zero-shot baseline is byte-for-byte reproducible."""
    import os, urllib.request
    if backend == "stub":
        return {"_stub": True, "text": ""}
    if backend == "gemini":
        return _ask_gemini(role_system, user, temperature)
    base = (os.environ.get("ONNES_LLM_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
            or "http://localhost:4000").rstrip("/")
    key = (os.environ.get("ONNES_LLM_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or "")
    model = os.environ.get("ONNES_CLAUDE_MODEL", "claude-opus-4.8")
    payload = {"model": model, "max_tokens": 4096,
               "messages": [{"role": "system", "content": role_system},
                            {"role": "user", "content": user}]}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read().decode())
        ch = body.get("choices") or []
        if not ch:
            return {"_error": "no choices"}
        text = (ch[0].get("message") or {}).get("content") or ""
        return A._extract_json(text) or {"_raw": text[:200]}
    except Exception as e:
        return {"_error": str(e)}


def _diagnose(user: str, backend: str, sc_samples: int) -> dict:
    """Run the Diagnostician once (sc_samples<=1) or with self-consistency voting.

    With sc_samples>1, sample the diagnosis sc_samples times at a diversity
    temperature and majority-vote the fault_class; return the modal sample's full
    reply, annotated with the vote tally. This is the cheap, reliable accuracy lever
    (self-consistency) — NOT a self-refine loop (which the literature shows hurts)."""
    sys_prompt = (
        "You are Diagnostician for a Bluefors dilution fridge. Stages temp1=50K, "
        "temp2=4K(magnet flange, quench risk), temp3=still, temp4=cold plate, "
        "temp5=mixing chamber(~10mK), temp6/7=magnet. Fault classes: " +
        ", ".join(CE.FAULT_CLASSES) + ". Reply ONLY JSON: "
        '{"fault_class": str, "severity": "none|low|medium|high", "reason": str}.')
    if sc_samples <= 1:
        return _ask(sys_prompt, user, backend)
    samples = [_ask(sys_prompt, user, backend, temperature=0.7) for _ in range(sc_samples)]
    votes = Counter(str(s.get("fault_class", "")).strip().lower()
                    for s in samples if isinstance(s, dict))
    votes.pop("", None)
    if not votes:
        return samples[0]
    winner = votes.most_common(1)[0][0]
    # return the first full reply whose class == the vote winner, tagged with the tally
    for s in samples:
        if str(s.get("fault_class", "")).strip().lower() == winner:
            return {**s, "_sc_votes": dict(votes), "_sc_samples": sc_samples}
    return {**samples[0], "_sc_votes": dict(votes), "_sc_samples": sc_samples}


# ---------------------------------------------------------------- verifier --
VERIFIER_SYS = (
    "You are Verifier, a skeptical second reader for a Bluefors dilution fridge. The panel "
    "concluded this window is NORMAL. Faint faults near the noise floor are exactly what a "
    "panel misses, so re-examine before agreeing. Stage map: temp3=still, temp4=cold plate, "
    "temp5=mixing chamber (~10 mK), flowmeter/p2/p5=flow & pressures.\n"
    "PHYSICS of the faults a 'normal' verdict can hide:\n"
    "- wiring_heat_ingress: a parasitic CONDUCTED load — cold plate (temp4) AND mixing "
    "chamber (temp5) BOTH drift UP together (even tens of mK), while still, flow, and all "
    "pressures stay FLAT. Correlated temp4+temp5 rise with flat flow/pressure = this fault, "
    "NOT normal (normal keeps temp4 and temp5 flat).\n"
    "- heat_load_spike: mixing chamber (temp5) rises but cold plate (temp4) stays flat "
    "(MXC-only). If temp4 also rises, it is wiring_heat_ingress, not heat_load_spike.\n"
    "- helium_leak / blocked_impedance: warm cold stages BUT flow drops and/or pressures "
    "move; if flow and pressures are flat it is not these.\n"
    "Be conservative: only overturn NORMAL when a specific physical pattern is present. If "
    "temperatures, flow, and pressures are genuinely flat, agree it is normal. "
    'Reply ONLY JSON: {"is_fault": bool, "fault_class": str, "reason": str}. '
    'Use fault_class from: ' + ", ".join(CE.FAULT_CLASSES) + '; use "normal" if truly normal.')


def run_verifier(cols: dict, backend: str = "litellm",
                 few_shot_block: str | None = None) -> dict:
    """Selective test-time self-verification: a skeptic re-reads a window the panel called
    NORMAL and can overturn it to a fault class. Physics-grounded (correlated cold-plate +
    mixing-chamber drift with flat flow/pressure = faint wiring_heat_ingress) and
    deliberately CONSERVATIVE — it fires only on 'normal' verdicts, so it is cheap and
    biased toward recovering missed faults, not manufacturing them.

    This is training-free test-time verification (SVSR, arXiv:2604.10228; physics-grounded
    self-verification in multi-agent science pipelines, arXiv:2606.18648; verifier-signal
    gating, arXiv:2607.02510), NOT self-refinement (which the literature shows hurts).
    Doubles as retry-hardening: if a mid-panel role dropped a transport call and the
    Supervisor defaulted to 'normal', the Verifier re-reads the raw window and recovers it."""
    summary = A.summarize_window(cols)
    s = json.dumps(summary, indent=2)
    user = ((f"{few_shot_block}\n\n" if few_shot_block else "") +
            f"Telemetry window the panel called NORMAL:\n{s}\n"
            "Re-examine for a faint fault hiding under a normal verdict.")
    return _ask(VERIFIER_SYS, user, backend)


def run_single_agent(cols: dict, backend: str = "litellm",
                     few_shot_block: str | None = None, sc_samples: int = 1) -> dict:
    """Single-call baseline for the panel-vs-single ablation (reviewers' 'Illusion of
    Multi-Agent Advantage' point). ONE LLM call does detection + classification directly,
    returning the SAME verdict shape the Supervisor emits, so it scores identically. The
    same few-shot / self-consistency levers apply, so this isolates the effect of the
    5-role decomposition itself, at 1/5 the calls."""
    summary = A.summarize_window(cols)
    s = json.dumps(summary, indent=2)
    sys_prompt = (
        "You are a dilution-fridge fault diagnostician. Stages temp1=50K, temp2=4K "
        "(magnet flange), temp3=still, temp4=cold plate, temp5=mixing chamber(~10mK), "
        "temp6/7=magnet. Fault classes: " + ", ".join(CE.FAULT_CLASSES) + ". In ONE step, "
        "decide if a fault is developing and which class. Reply ONLY JSON: "
        '{"fault_detected": bool, "fault_class": str, "confidence": str}.')
    user = (f"{few_shot_block}\n\n" if few_shot_block else "") + \
        f"Telemetry window summary:\n{s}\nDiagnose."
    if sc_samples > 1:
        samples = [_ask(sys_prompt, user, backend, temperature=0.7) for _ in range(sc_samples)]
        votes = Counter(str(x.get("fault_class", "")).strip().lower()
                        for x in samples if isinstance(x, dict))
        votes.pop("", None)
        verdict = None
        if votes:
            win = votes.most_common(1)[0][0]
            verdict = next((x for x in samples
                            if str(x.get("fault_class", "")).strip().lower() == win), samples[0])
            verdict = {**verdict, "_sc_votes": dict(votes)}
        else:
            verdict = samples[0]
    else:
        verdict = _ask(sys_prompt, user, backend)
    # present as a one-role "panel" so downstream scoring (_panel_verdict) is unchanged
    return {"supervisor": verdict}


def run_panel(cols: dict, backend: str = "litellm",
              few_shot_block: str | None = None, sc_samples: int = 1) -> dict:
    """Run the multi-agent panel on one telemetry window. Returns each role's output.

    few_shot_block (optional): labeled contrastive examples prepended to the
    Diagnostician's prompt. sc_samples (optional, default 1): self-consistency vote
    count for the Diagnostician. Both default to the original zero-shot single-sample
    behavior so the baseline reproduces exactly."""
    summary = A.summarize_window(cols)
    s = json.dumps(summary, indent=2)
    out = {}

    # 1. Sentinel
    out["sentinel"] = _ask(
        "You are Sentinel, watching dilution-fridge telemetry. Reply ONLY JSON: "
        '{"anomaly_developing": bool, "which_channels": [str], "confidence": "low|med|high"}.',
        f"Telemetry window summary:\n{s}\nIs an anomaly developing?", backend)

    # 2. Diagnostician (optionally few-shot + self-consistency)
    diag_user = (f"{few_shot_block}\n\n" if few_shot_block else "") + \
        f"Telemetry:\n{s}\nClassify."
    out["diagnostician"] = _diagnose(diag_user, backend, sc_samples)

    # 3. Operator
    out["operator"] = _ask(
        "You are Operator. Propose ONE corrective action for the diagnosed fault. "
        'Reply ONLY JSON: {"action": str, "urgency": "low|medium|high"}.',
        f"Telemetry:\n{s}\nDiagnosis: {json.dumps(out['diagnostician'])}\nAction?", backend)

    # 4. Guardian
    out["guardian"] = _ask(
        "You are Guardian, enforcing fridge safety. Block actions that could quench the "
        "magnet, warm the mixing chamber uncontrollably, or exceed heater limits. "
        'Reply ONLY JSON: {"approved": bool, "why": str}.',
        f"Proposed action: {json.dumps(out['operator'])}\nApprove?", backend)

    # 5. Supervisor
    out["supervisor"] = _ask(
        "You are Supervisor. Reconcile the panel into a final verdict. Reply ONLY JSON: "
        '{"fault_detected": bool, "fault_class": str, "final_action": str, "confidence": str}.',
        f"Sentinel:{json.dumps(out['sentinel'])}\nDiagnostician:{json.dumps(out['diagnostician'])}\n"
        f"Operator:{json.dumps(out['operator'])}\nGuardian:{json.dumps(out['guardian'])}\nFinal verdict?",
        backend)
    return out

