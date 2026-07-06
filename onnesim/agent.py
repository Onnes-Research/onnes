"""
CryoAgent — an LLM-agent operations layer for dilution-refrigerator telemetry.

This is the piece the FRTMS state-of-the-art is missing: FRTMS (arXiv:2602.05160)
monitors and sends THRESHOLD + LAG alerts (Grafana/Slack). No ML, no agent, no
diagnosis. CryoAgent reads a telemetry window and returns a structured judgment:

    { "fault_detected": bool,
      "fault_class": <one of faults.FAULT_CLASSES>,
      "severity": "none|low|medium|high",
      "recommended_action": str,
      "rationale": str }

Two backends, chosen automatically:
  - "claude"  : real Anthropic API call, if ANTHROPIC_API_KEY is set and the SDK
                is installed. This is the backend used for paper results.
  - "stub"    : a transparent, rule-based fallback so the pipeline is runnable
                offline. Its outputs are clearly labeled backend="stub" and must
                NEVER be reported as model performance.

Design note: the agent sees a COMPACT NUMERIC SUMMARY of the window (down-sampled
channel stats), not raw 5000-row CSVs — mirroring how you'd really feed telemetry
to an LLM within a context budget.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from . import faults as F

MODEL_DEFAULT = "claude-opus-4.8"  # override via ONNES_CLAUDE_MODEL


# --------------------------------------------------------------------------- #
# Feature extraction: turn a raw telemetry window into a small text summary.
# --------------------------------------------------------------------------- #
def summarize_window(cols: dict, n_buckets: int = 6) -> dict:
    """Compact, LLM-friendly summary of a telemetry window.

    Reports, per key channel: start value, end value, % change, and a coarse
    down-sampled trajectory. Keeps the prompt small and model-agnostic.
    """
    t = cols["t_s"]
    dur_h = float((t[-1] - t[0]) / 3600.0)
    channels = ["temp1_T", "temp2_T", "temp3_T", "temp4_T", "temp5_T",
                "temp6_T", "temp7_T", "flowmeter", "p2", "p5"]
    summary = {"duration_h": round(dur_h, 2), "channels": {}}
    for ch in channels:
        if ch not in cols:
            continue
        x = np.asarray(cols[ch], dtype=float)
        start, end = float(x[0]), float(x[-1])
        pct = (end - start) / (abs(start) + 1e-12) * 100.0
        buckets = [float(np.mean(b)) for b in np.array_split(x, n_buckets)]
        summary["channels"][ch] = {
            "start": round(start, 5),
            "end": round(end, 5),
            "pct_change": round(pct, 1),
            "trajectory": [round(v, 5) for v in buckets],
        }
    return summary


# --------------------------------------------------------------------------- #
# Agent result type
# --------------------------------------------------------------------------- #
@dataclass
class AgentVerdict:
    fault_detected: bool
    fault_class: str
    severity: str
    recommended_action: str
    rationale: str
    backend: str

    def as_dict(self):
        return asdict(self)


SYSTEM_PROMPT = (
    "You are CryoAgent, an operations agent for a Bluefors LD400 dilution "
    "refrigerator used in a quantum-computing / low-temperature physics lab. "
    "You receive a compact numeric summary of a telemetry window. The fridge "
    "cools in stages: temp1=50K, temp2=4K flange (a 9T magnet is thermalized "
    "here; watch for quench = magnet self-heating and dumping load onto 4K), "
    "temp3=still(~0.7K), temp4=cold plate(~50mK), temp5=mixing chamber(~10mK), "
    "temp6/temp7=magnet sensors, flowmeter=mixture circulation, p2/p5=GHS "
    "pressures. Possible fault classes: " + ", ".join(F.FAULT_CLASSES) + ". "
    "Decide whether a fault is developing, classify it, rate severity, and give "
    "ONE concrete recommended action. Respond ONLY with a JSON object with keys: "
    "fault_detected (bool), fault_class (string from the list), severity "
    "(none|low|medium|high), recommended_action (string), rationale (string)."
)


def _prompt_for(summary: dict) -> str:
    return (
        "Telemetry window summary (JSON):\n"
        + json.dumps(summary, indent=2)
        + "\n\nReturn only the JSON verdict."
    )


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _verdict_from_text(text: str, backend_label: str) -> Optional[AgentVerdict]:
    """Shared: parse model text -> validated AgentVerdict (or None if unparseable)."""
    data = _extract_json(text)
    if data is None:
        return None
    # Validate against known label sets; coerce hallucinated values to safe
    # defaults so an invalid class can't silently corrupt scoring.
    fclass = str(data.get("fault_class", "normal"))
    if fclass not in F.FAULT_CLASSES:
        fclass = "normal"
    sev = str(data.get("severity", "none"))
    if sev not in ("none", "low", "medium", "high"):
        sev = "none"
    return AgentVerdict(
        fault_detected=bool(data.get("fault_detected", False)),
        fault_class=fclass,
        severity=sev,
        recommended_action=str(data.get("recommended_action", "")),
        rationale=str(data.get("rationale", "")),
        backend=backend_label,
    )


def _call_claude(summary: dict, model: str) -> Optional[AgentVerdict]:
    """Real Anthropic call. Returns None if unavailable so caller can fall back."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _prompt_for(summary)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return _verdict_from_text(text, "claude:" + model)
    except Exception as e:
        # Never crash the pipeline on an API hiccup; report and fall back.
        print(f"[CryoAgent] Claude call failed ({e}); falling back to stub.")
        return None


def _call_litellm(summary: dict, model: str) -> Optional[AgentVerdict]:
    """Call an OpenAI-compatible endpoint (e.g. a local LiteLLM proxy) via stdlib
    urllib — no extra dependency. Config from env, with sensible local defaults:

        ONNES_LLM_BASE_URL   (default $ANTHROPIC_BASE_URL or http://localhost:4000)
        ONNES_LLM_API_KEY    (default $ANTHROPIC_AUTH_TOKEN or $OPENAI_API_KEY)
        ONNES_CLAUDE_MODEL / model arg   (e.g. claude-opus-4.8)

    Returns None on any failure so the caller can fall back to the stub.
    """
    import json as _json
    import urllib.request

    base = (os.environ.get("ONNES_LLM_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or "http://localhost:4000").rstrip("/")
    key = (os.environ.get("ONNES_LLM_API_KEY")
           or os.environ.get("ANTHROPIC_AUTH_TOKEN")
           or os.environ.get("OPENAI_API_KEY")
           or "")
    url = base + "/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": 4096,  # reasoning models can spend a lot before the final JSON
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt_for(summary)},
        ],
    }
    req = urllib.request.Request(
        url, data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            fr = "unknown"
            print(f"[CryoAgent] LiteLLM returned no choices "
                  f"(usage={body.get('usage')}); falling back to stub.")
            return None
        text = (choices[0].get("message") or {}).get("content") or ""
        return _verdict_from_text(text, "litellm:" + model)
    except Exception as e:
        print(f"[CryoAgent] LiteLLM call failed ({e}); falling back to stub.")
        return None


def _balanced_json_candidates(text: str):
    """Yield every balanced {...} substring, so a stray/decoy brace before the
    real object doesn't corrupt extraction (proven bug: first-brace..last-brace
    spans prose between two objects). Scans left-to-right, shortest-valid-first."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]


def _extract_json(text: str) -> Optional[dict]:
    """Robustly pull a JSON object from an LLM response.

    Order: (1) a ```json ...``` fenced block, (2) any balanced {...} candidate
    that parses. Returns None if nothing parses (caller degrades to stub or
    raises, per backend) — never fabricates.
    """
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for candidate in _balanced_json_candidates(text):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _stub_verdict(summary: dict) -> AgentVerdict:
    """Transparent rule-based baseline agent. NOT a model. Clearly labeled."""
    ch = summary["channels"]
    mxc = ch.get("temp5_T", {})
    still = ch.get("temp3_T", {})
    flow = ch.get("flowmeter", {})
    magnet = ch.get("temp6_T", {})
    four_k = ch.get("temp2_T", {})

    detected, fclass, sev, action, why = False, "normal", "none", "continue monitoring", "all channels nominal"

    # Magnet quench: 4K/magnet rising sharply — highest priority.
    if magnet.get("pct_change", 0) > 30 or four_k.get("pct_change", 0) > 25:
        detected, fclass, sev = True, "magnet_quench", "high"
        action = "pause field ramp; stop injecting current; investigate 4K flange load"
        why = "magnet/4K temperature rising sharply — quench risk"
    # Circulation drop -> blocked impedance.
    elif flow.get("pct_change", 0) < -25:
        detected, fclass, sev = True, "blocked_impedance", "medium"
        action = "check impedance / flow restriction in the still line"
        why = "mixture flow dropped substantially"
    # Cold stages warming -> helium leak / heat load.
    elif mxc.get("pct_change", 0) > 40 or still.get("pct_change", 0) > 40:
        detected, fclass, sev = True, "helium_leak", "medium"
        action = "inspect mixture inventory and cold-stage heat loads"
        why = "cold stages warming without flow change"

    return AgentVerdict(detected, fclass, sev, action, why, backend="stub")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_agent(cols: dict, backend: str = "auto", model: Optional[str] = None) -> AgentVerdict:
    """Run CryoAgent on one telemetry window.

    backend:
      "auto"    — try LiteLLM proxy, then Anthropic SDK, else stub.
      "litellm" — OpenAI-compatible endpoint (local LiteLLM proxy). Errors if unreachable.
      "claude"  — Anthropic SDK. Errors if unavailable.
      "stub"    — transparent rule-based fallback (NOT a model).
    """
    model = model or os.environ.get("ONNES_CLAUDE_MODEL", MODEL_DEFAULT)
    summary = summarize_window(cols)

    if backend in ("auto", "litellm"):
        v = _call_litellm(summary, model)
        if v is not None:
            return v
        if backend == "litellm":
            raise RuntimeError(
                "backend='litellm' requested but the OpenAI-compatible endpoint was "
                "unreachable/unparseable. Check the proxy at ONNES_LLM_BASE_URL / "
                "ANTHROPIC_BASE_URL (default http://localhost:4000)."
            )

    if backend in ("auto", "claude"):
        v = _call_claude(summary, model)
        if v is not None:
            return v
        if backend == "claude":
            raise RuntimeError(
                "backend='claude' requested but ANTHROPIC_API_KEY / anthropic SDK "
                "unavailable. Install `anthropic` and set the key, or use backend='stub'."
            )
    return _stub_verdict(summary)
