"""
Offline mock of the Anthropic Python SDK for testing CryoAgent's parsing path.

Why this exists
---------------
`onnesim/agent.py` talks to Claude through exactly one call shape:

    import anthropic
    client = anthropic.Anthropic(api_key=...)
    msg = client.messages.create(model=..., max_tokens=..., system=..., messages=...)
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

...and then feeds `text` into `agent._extract_json(...)` and builds an
`AgentVerdict`. To *prove* that parsing + verdict construction is trustworthy
without spending API tokens, we need to feed KNOWN model outputs through that
exact code path. This module supplies:

  * `MockAnthropic` — a drop-in replacement for `anthropic.Anthropic` whose
    `.messages.create(...)` returns an object mimicking the real return shape:
    a message whose `.content` is a list of blocks, each with `.type == "text"`
    and a `.text` attribute (non-text blocks are supported too, so we exercise
    the `getattr(b, "type", "") == "text"` filter).

  * `mock_anthropic(script)` — a context manager that injects a fake `anthropic`
    module into `sys.modules` and sets a dummy `ANTHROPIC_API_KEY`, so the REAL
    `agent._call_claude` runs against the mock. agent.py is NOT modified.

  * `run_agent_with_mock(cols, scripted_text, backend="claude")` — the headline
    helper: drive `agent.run_agent` on a telemetry window with a scripted model
    response, exercising `_extract_json` + `AgentVerdict` construction.

  * `SCRIPTS` — canonical scripted responses covering clean JSON, prose-wrapped
    JSON, markdown-fenced JSON, a stray-brace decoy, malformed JSON, no JSON at
    all, empty output, and an API-hiccup (raised exception).

Nothing here imports the real `anthropic` SDK, so it runs fully offline.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from typing import Any, Callable, Iterable, Optional, Union

from . import agent as A


# --------------------------------------------------------------------------- #
# Return-shape mimicry: blocks -> message
# --------------------------------------------------------------------------- #
class MockTextBlock:
    """Mimics an Anthropic `TextBlock`: has `.type == "text"` and `.text`."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MockTextBlock({self.text!r})"


class MockNonTextBlock:
    """Mimics a non-text block (e.g. a `tool_use` block).

    Deliberately has NO usable `.text` so we can prove agent.py's
    `getattr(b, "type", "") == "text"` filter skips it instead of crashing.
    """

    def __init__(self, block_type: str = "tool_use"):
        self.type = block_type

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MockNonTextBlock({self.type!r})"


class MockMessage:
    """Mimics the object returned by `client.messages.create(...)`."""

    def __init__(self, blocks: list):
        self.content = blocks
        self.role = "assistant"
        self.stop_reason = "end_turn"
        self.model = "mock-claude"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MockMessage(content={self.content!r})"


# A "response spec" is one of:
#   * str                       -> a single text block containing that string
#   * list[...]                 -> multiple blocks; each item is either a str
#                                  (text block) or a MockNonTextBlock instance
#   * Exception instance        -> create() RAISES it (simulate an API hiccup)
#   * MockMessage               -> returned verbatim
ResponseSpec = Union[str, list, Exception, MockMessage]


def _blocks_from_spec(spec: ResponseSpec) -> MockMessage:
    """Turn a response spec into a MockMessage (mimicking real return shape)."""
    if isinstance(spec, MockMessage):
        return spec
    if isinstance(spec, str):
        return MockMessage([MockTextBlock(spec)])
    if isinstance(spec, list):
        blocks = []
        for item in spec:
            if isinstance(item, str):
                blocks.append(MockTextBlock(item))
            elif isinstance(item, (MockTextBlock, MockNonTextBlock)):
                blocks.append(item)
            else:
                raise TypeError(f"unsupported block item: {item!r}")
        return MockMessage(blocks)
    raise TypeError(f"unsupported response spec: {spec!r}")


# --------------------------------------------------------------------------- #
# The mock client
# --------------------------------------------------------------------------- #
class _MockMessages:
    def __init__(self, parent: "MockAnthropic"):
        self._parent = parent

    def create(self, **kwargs) -> MockMessage:
        parent = self._parent
        parent.calls.append(kwargs)  # record for assertions
        spec = parent._next_spec()
        if isinstance(spec, Exception):
            raise spec  # simulate an API error; agent._call_claude catches it
        return _blocks_from_spec(spec)


class MockAnthropic:
    """Drop-in stand-in for `anthropic.Anthropic`.

    Because `agent._call_claude` constructs a NEW client on every call, the
    scripted responses live on the *class* (a shared queue), so a sequence of
    scripted responses survives across successive `run_agent` invocations.
    """

    # Shared, process-global script queue + call log (see note above).
    _script: list = []
    _script_repeat_last: bool = True
    calls: list = []

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.messages = _MockMessages(self)

    # -- scripting -------------------------------------------------------- #
    @classmethod
    def set_script(cls, script: Union[ResponseSpec, Iterable[ResponseSpec]],
                   repeat_last: bool = True) -> None:
        """Load the response(s) create() will yield, in order.

        A single spec (str/list/Exception/MockMessage) is treated as one
        response reused for every call. An iterable of specs is a FIFO queue.
        """
        if isinstance(script, (str, list, Exception, MockMessage)):
            cls._script = [script]
        else:
            cls._script = list(script)
        cls._script_repeat_last = repeat_last
        cls.calls = []

    def _next_spec(self) -> ResponseSpec:
        script = type(self)._script
        if not script:
            raise RuntimeError("MockAnthropic has no scripted response set.")
        if len(script) == 1:
            return script[0] if type(self)._script_repeat_last else script.pop(0)
        return script.pop(0)


def make_fake_anthropic_module() -> types.ModuleType:
    """Build a fake `anthropic` module exposing `Anthropic` = MockAnthropic."""
    mod = types.ModuleType("anthropic")
    mod.Anthropic = MockAnthropic

    class APIError(Exception):
        ...

    class APIConnectionError(APIError):
        ...

    mod.APIError = APIError
    mod.APIConnectionError = APIConnectionError
    return mod


# --------------------------------------------------------------------------- #
# Injection: run the REAL agent against the mock, offline
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def mock_anthropic(script: Union[ResponseSpec, Iterable[ResponseSpec]],
                   api_key: str = "mock-key-not-real",
                   repeat_last: bool = True):
    """Context manager: make `agent._call_claude` run against MockAnthropic.

    Injects a fake `anthropic` module into `sys.modules` and sets a dummy
    ANTHROPIC_API_KEY for the duration, then restores prior state on exit
    (important so parallel/other imports in the same process are unaffected).
    """
    MockAnthropic.set_script(script, repeat_last=repeat_last)

    saved_mod = sys.modules.get("anthropic", None)
    saved_mod_present = "anthropic" in sys.modules
    saved_key = os.environ.get("ANTHROPIC_API_KEY", None)
    # Neutralize the LiteLLM proxy backend so `backend="auto"` tests are HERMETIC:
    # without this, a live proxy on localhost:4000 would answer before the mocked
    # anthropic path and the offline self-test would depend on external state.
    saved_litellm = A._call_litellm

    sys.modules["anthropic"] = make_fake_anthropic_module()
    os.environ["ANTHROPIC_API_KEY"] = api_key
    A._call_litellm = lambda summary, model: None
    try:
        yield MockAnthropic
    finally:
        A._call_litellm = saved_litellm
        # Restore anthropic module slot exactly as it was.
        if saved_mod_present:
            sys.modules["anthropic"] = saved_mod
        else:
            sys.modules.pop("anthropic", None)
        # Restore the API key env var exactly as it was.
        if saved_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved_key


def run_agent_with_mock(cols: dict, scripted_text: ResponseSpec,
                        backend: str = "claude",
                        model: Optional[str] = None) -> A.AgentVerdict:
    """Drive `agent.run_agent` on `cols` with a scripted model response.

    This exercises agent.py's real model path end-to-end:
        summarize_window -> _call_claude (mock) -> _extract_json -> AgentVerdict

    backend semantics (unchanged from agent.py):
      * "claude": pure model path. If the scripted text yields no parseable
        JSON, `_call_claude` returns None and `run_agent` RAISES RuntimeError
        (proves the parser rejects junk rather than inventing a verdict).
      * "auto":   model path with graceful fallback. Unparseable output falls
        back to the transparent stub verdict (proves no crash).

    Returns the resulting AgentVerdict. Use `MockAnthropic.calls` afterwards to
    assert the API was actually invoked (i.e. we did not silently hit the stub).
    """
    with mock_anthropic(scripted_text):
        return A.run_agent(cols, backend=backend, model=model)


def call_claude_with_mock(cols: dict, scripted_text: ResponseSpec,
                          model: Optional[str] = None) -> Optional[A.AgentVerdict]:
    """Lower-level: run the real `agent._call_claude` against the mock.

    Returns the AgentVerdict, or None when parsing fails (mirrors _call_claude).
    Lets a test observe the None-on-malformed contract without run_agent's raise.
    """
    model = model or A.MODEL_DEFAULT
    summary = A.summarize_window(cols)
    with mock_anthropic(scripted_text):
        return A._call_claude(summary, model)


# --------------------------------------------------------------------------- #
# Canonical scripted responses (clean / wrapped / malformed / none)
# --------------------------------------------------------------------------- #
def verdict_json(fault_detected: bool = False, fault_class: str = "normal",
                 severity: str = "none", recommended_action: str = "continue monitoring",
                 rationale: str = "all channels nominal") -> str:
    """Build a clean JSON verdict string (what a well-behaved model returns)."""
    return json.dumps({
        "fault_detected": fault_detected,
        "fault_class": fault_class,
        "severity": severity,
        "recommended_action": recommended_action,
        "rationale": rationale,
    })


_QUENCH = verdict_json(True, "magnet_quench", "high",
                       "pause field ramp; stop injecting current; inspect 4K flange load",
                       "magnet/4K temperature rising sharply — quench risk")
_NORMAL = verdict_json(False, "normal", "none", "continue monitoring",
                       "all channels within nominal band")

# A dict of ready-made scripts keyed by the *kind of model output* they emulate.
SCRIPTS: dict = {
    # 1) Clean JSON, exactly as the system prompt requests.
    "clean_quench": _QUENCH,
    "clean_normal": _NORMAL,

    # 2) JSON embedded in conversational prose.
    "prose_quench": (
        "Looking at the telemetry, the magnet sensor is climbing fast. "
        "Here is my verdict:\n" + _QUENCH + "\nLet me know if you need more detail."
    ),

    # 3) JSON inside a markdown ```json fenced code block.
    "fenced_normal": (
        "Everything looks stable. Verdict below:\n\n```json\n" + _NORMAL + "\n```\n"
    ),
    "fenced_quench": (
        "Alert — see the fenced JSON:\n\n```json\n" + _QUENCH + "\n```"
    ),

    # 4) Valid verdict preceded by a stray brace (a KNOWN weakness of the
    #    naive first-'{' / last-'}' extractor; kept so tests document it).
    "stray_brace_quench": (
        "Consider the affected channels {temp2_T, temp6_T}. My verdict is: " + _QUENCH
    ),

    # 5) Multi-block message: a non-text block followed by a text block with
    #    clean JSON (exercises the type=="text" filter + the join).
    "multiblock_normal": [MockNonTextBlock("tool_use"), _NORMAL],

    # 6) Malformed JSON — looks JSON-ish but will not parse.
    "malformed": "{fault_detected: yes, fault_class: magnet_quench, severity: high}",

    # 7) No JSON at all — plain prose.
    "no_json": "The fridge appears nominal; continue routine monitoring.",

    # 8) Empty model output.
    "empty": "",

    # 9) Simulated API hiccup: create() raises; agent must fall back, not crash.
    "api_error": RuntimeError("simulated transient API error"),
}


__all__ = [
    "MockTextBlock", "MockNonTextBlock", "MockMessage", "MockAnthropic",
    "make_fake_anthropic_module", "mock_anthropic",
    "run_agent_with_mock", "call_claude_with_mock",
    "verdict_json", "SCRIPTS",
]
