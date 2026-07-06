# Onnes: A Physics-Grounded Multi-Agent LLM Simulator for Cryogenic Fault Diagnosis in Quantum Computing Infrastructure

Onnes is a physics-grounded digital-twin simulator of a dilution refrigerator that drives a
live multi-agent large-language-model (LLM) operations layer. It is also the harness for a
controlled head-to-head comparison between a zero-shot LLM agent panel and a supervised ML
classifier on cryogenic fault diagnosis.

Dilution refrigerators are the enabling infrastructure of superconducting quantum computers,
yet their fault diagnosis is still dominated by threshold alarms that report *that* something
is wrong, not *what*. Onnes tests whether a reasoning LLM panel can do the "what", and
measures it against supervised ML.

Code and released run logs: https://github.com/Onnes-Research/onnes

## Headline result

Every number below is drawn from a released artifact under `outputs/`; none are
hand-authored. See [Reproducibility](#reproducibility).

| Task | Zero-shot agent panel | + few-shot + self-consistency | Supervised ML |
|---|---|---|---|
| Detection (fault vs normal) | 0.965 (no significant difference vs ML, McNemar *p* = 0.07) | — | 0.995 |
| Classification (6-way) | 0.685 (errors concentrate on the engineered confusable faults) | 0.990 | 0.985 (RF), 0.995 (TabPFN-2.5) |

The result is not "agents win." Zero-shot, the panel ties ML on detection and trails it on
classification. Curated contrastive few-shot demonstrations and self-consistency voting then
raise classification from 0.685 to 0.990, which matches the supervised model (0.985) with no
parameter updates and six labeled demonstrations. An ablation attributes the gain almost
entirely to the demonstrations rather than the multi-agent structure. Run as a continuous
monitor across a nine-run fault×seed sweep, the agent catches every developing fault within
one poll interval, and a confidence gate suppresses pre-onset false alarms. That false-alarm
rate is backend-dependent: 0 in every cell of the Gemini 3.1 Pro sweep, versus 11 in the
single Claude/Opus run.

As a first sim-to-real check, a detector trained only on real BlueFors healthy telemetry has
a real-hardware false-alarm rate of 6.4% and 100% recall on physics faults injected onto real
held-out windows, so its low false-alarm rate does not trade off against missed faults.

The agent's value is therefore not raw accuracy. It is interpretable physics-grounded
reasoning, a per-role audit trail, and reaching supervised accuracy from 6 curated examples
instead of 300 labeled scenarios.

## What's in the twin

- 5-stage lumped thermal model (`onnesim/cryo_engine.py`) with a *T²* dilution-cooling floor
  (`dilution_cooling.py`) and Cryowala-calibrated heat loads (`cryowala_physics.py`).
- Real-fridge noise fingerprint learned from BlueFors logs (`virtual_clone.py`,
  `bluefors_data.py`): per-stage relative noise and cross-stage correlation, not toy Gaussian.
- Six physics-grounded fault classes (`realistic_faults.py`); three overlap on temperature but
  separate on flow and pressure (the hard cases).

## What's in the agent layer

- 5-role panel (`onnesim/multi_agent.py`): Sentinel, Diagnostician, Operator, Guardian,
  Supervisor. Each is a real LLM call.
- In-context levers: curated contrastive few-shot and self-consistency voting. We deliberately
  avoid debate and self-refine, based on 2026 evidence that their gains are largely a
  test-time-compute artifact.
- Backends (`_ask` in `multi_agent.py`): `litellm` (Claude via an OpenAI-compatible proxy),
  `gemini` (Google GenAI SDK), and `stub` (offline).
- Statistics (`onnesim/stats.py`): Clopper–Pearson CIs and exact paired McNemar, recomputed
  from the released turn logs.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline, no API key needed (stub backend):
python scripts/selftest_agent.py                 # sanity-check the panel end-to-end
python scripts/run_benchmark.py                  # ML baselines on the realistic engine

# Live agent evaluation (needs an LLM backend; keys via env vars only):
#   litellm/Claude: ONNES_LLM_BASE_URL, ONNES_LLM_API_KEY, ONNES_CLAUDE_MODEL
#   gemini:         GEMINI_API_KEY, ONNES_GEMINI_MODEL
python scripts/eval_gemini.py --n 200            # cross-backend replication
```

## Reproducibility

Every number in this repo maps to a released artifact under `outputs/` (results as `*.json`,
the full per-turn audit trail as `*.jsonl`). Statistical tests are recomputed from the turn
logs by `onnesim/stats.py`; they are not stored numbers.

| Claim | Source artifact |
|---|---|
| Zero-shot 0.965 / 0.685 (n=200) | `outputs/agent_eval_results.json` |
| Head-to-head CIs + McNemar | `outputs/head_to_head_stats.json` |
| Enhanced lift (n=24) | `outputs/technique_lift.json` |
| Enhanced panel (n=200), 0.990 | `outputs/agent_eval_fewshot_n200_results.json` |
| Ablation (levers + architecture) | `outputs/ablation_results.json` |
| Baseline zoo (RF, HGB, LightGBM, TabPFN-2.5) | `outputs/baseline_zoo.json` |
| Cost / latency | `outputs/cost_model.json` |
| Continuous monitor (latency, gating) | `outputs/continuous_monitor.json`, `outputs/monitor_gating.json` |
| Monitor sweep (9 runs, Gemini) | `outputs/monitor_sweep_gemini.json` |
| ML stress macro-F₁ | `outputs/benchmark_results.json` |
| Cross-backend (Claude vs Gemini) | `outputs/gemini_vs_claude.json` |
| Sim-to-real detector (6.4% real FA, 100% recall) | `outputs/real_detect.json` |

Regenerate the figures from the artifacts:

```bash
python scripts/make_figures.py     # writes outputs/figures/*.png
```

## Scope and limitations

- The twin is a forward physics model with a learned noise fingerprint, not a
  hardware-coupled bidirectional twin. It is fingerprinted to real BlueFors logs, but
  validation against labeled real-fridge fault episodes is the ultimate test and is not yet
  done.
- The head-to-head is on simulated telemetry. The 6-vs-300-example gap is real on the twin;
  whether the economic case (fewer labels) transfers depends on closing the sim-to-real gap.
- Detection latency in the monitor is cadence-bounded by the poll interval. It is a detection
  latency set by the schedule, not a physical lead time.

## Repository layout

```
onnesim/     simulator + agent layer + stats (the importable package)
scripts/     runnable entry points (eval_*, run_*, make_figures, train_*)
outputs/     released run logs + benchmark artifacts (every number)
tests/       test suite
```

## License

MIT (see `LICENSE`).
