#!/usr/bin/env python3
"""
run_stress_live.py — the FULL extensive stress + reliability battery on a LIVE backend.

Runs every harness in onnesim.agent_stress + agent_passk against the real multi-agent panel
(default backend: gemini / Gemini 3.1 Pro via the Google GenAI SDK), checkpointing each
harness's result to outputs/ as it finishes so a multi-hour run survives interruption.

This is the LIVE counterpart to the zero-API stub smoke run. It is opt-in and prints a call
budget before spending. Nothing here is re-implemented: it wires the existing harnesses to
onnesim.agent_stress.panel_predict, which is the real run_panel.

Usage (key via env, never hard-coded):
    GEMINI_API_KEY=... .venv/bin/python scripts/run_stress_live.py --backend gemini \
        --n 40 --passk-n 30 --passk-k 5 --workers 3

    # resume: each harness writes its own JSON; already-present ones are skipped with --resume
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import agent_stress as ST
from onnesim import agent_passk as PK


def _threaded_predict(predict_fn, workers: int):
    """Wrap a predict_fn so a harness's per-scenario loop can run scenarios concurrently.

    The harnesses call predict_fn one scenario at a time (serial); to use Gemini's allowed
    concurrency we instead pre-warm nothing and rely on the harness loops being independent.
    For simplicity and rate-safety we keep predict_fn serial here but cap workers at the
    panel level via the run_panel threadpool. (Kept explicit so the knob is visible.)
    """
    return predict_fn


def _done(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini", choices=["gemini", "litellm", "stub"])
    ap.add_argument("--n", type=int, default=40, help="scenarios per stress family")
    ap.add_argument("--adv-n", type=int, default=15, help="fault scenarios for adversarial")
    ap.add_argument("--passk-n", type=int, default=30, help="scenarios for pass^k")
    ap.add_argument("--passk-k", type=int, default=5, help="repeats for pass^k")
    ap.add_argument("--sc", type=int, default=1, help="self-consistency votes in the panel")
    ap.add_argument("--workers", type=int, default=3, help="(reserved) panel-level concurrency")
    ap.add_argument("--resume", action="store_true", help="skip harnesses already on disk")
    ap.add_argument("--only", default="", help="comma list: metamorphic,perturbation,fault,adversarial,passk")
    args = ap.parse_args()

    tag = args.backend
    if args.backend == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("[live] set GEMINI_API_KEY first."); sys.exit(1)

    predict = (ST.stub_predict if args.backend == "stub"
               else ST.panel_predict(backend=args.backend, sc_samples=args.sc))

    # Set the module-level concurrency knob so every harness runs scenarios in parallel.
    # Gemini is RPM-limited, so keep this modest (3-5); the panel's 5 roles stay serial
    # within a scenario (they depend on each other), but scenarios run concurrently.
    ST.STRESS_WORKERS = max(1, args.workers)
    print(f"[live] scenario-level concurrency: {ST.STRESS_WORKERS} workers")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    def want(name): return not only or name in only

    # ---- call budget estimate (5 roles/panel; +sc-1 on the diagnostician) ----
    per_panel = 5 + (args.sc - 1)
    est = 0
    if want("metamorphic"): est += args.n * (1 + len(ST.METAMORPHIC_RELATIONS)) * per_panel
    if want("perturbation"): est += args.n * 4 * per_panel
    if want("fault"):        est += args.n * 3 * per_panel
    if want("adversarial"):  est += args.adv_n * (1 + len(ST.INJECTION_STRINGS)) * per_panel
    if want("passk"):        est += args.passk_n * args.passk_k * per_panel
    print(f"[live] backend={args.backend}  est. LLM calls ~= {est:,}  "
          f"(at ~{per_panel} calls/panel, ~10-20s each)")
    print(f"[live] families: " + ", ".join(n for n in
          ["metamorphic","perturbation","fault","adversarial","passk"] if want(n)))

    t0 = time.time()
    results = {}

    if want("metamorphic"):
        p = "outputs/live_metamorphic.json"
        if args.resume and _done(p):
            print("[live] metamorphic: resume-skip")
        else:
            print("\n[live] === METAMORPHIC (label-free verdict stability) ===")
            r = ST.run_metamorphic(predict, n=args.n)
            json.dump(r, open(p, "w"), indent=2); results["metamorphic"] = r
            print(f"[live] metamorphic stability = {r['overall_metamorphic_stability']} -> {p}")

    if want("perturbation"):
        p = "outputs/live_perturbation.json"
        if args.resume and _done(p):
            print("[live] perturbation: resume-skip")
        else:
            print("\n[live] === PERTURBATION (accuracy vs sensor-noise eps) ===")
            r = ST.run_perturbation(predict, n=args.n)
            json.dump(r, open(p, "w"), indent=2); results["perturbation"] = r
            print(f"[live] degradation clean->maxeps = {r['degradation_clean_to_max_eps']} -> {p}")

    if want("fault"):
        p = "outputs/live_fault_injection.json"
        if args.resume and _done(p):
            print("[live] fault_injection: resume-skip")
        else:
            print("\n[live] === FAULT-INJECTION (truncated-telemetry chaos, lambda) ===")
            r = ST.run_fault_injection(lambda _l: predict, n=args.n)
            json.dump(r, open(p, "w"), indent=2); results["fault_injection"] = r
            print(f"[live] graceful under truncation = {r['graceful']} -> {p}")

    if want("adversarial"):
        p = "outputs/live_adversarial.json"
        if args.resume and _done(p):
            print("[live] adversarial: resume-skip")
        else:
            print("\n[live] === ADVERSARIAL INJECTION (prompt-injection via data channel) ===")
            r = ST.run_adversarial(predict, n=args.adv_n)
            json.dump(r, open(p, "w"), indent=2); results["adversarial"] = r
            print(f"[live] mean attack success rate = {r['mean_attack_success_rate']} -> {p}")

    if want("passk"):
        p = "outputs/live_passk.json"
        if args.resume and _done(p):
            print("[live] passk: resume-skip")
        else:
            print(f"\n[live] === PASS^{args.passk_k} (deployability reliability) ===")
            r = PK.run(predict, n=args.passk_n, k=args.passk_k, tag=tag, out_path=p,
                       workers=max(1, args.workers))
            results["passk"] = r

    elapsed = round(time.time() - t0, 1)
    summary = {
        "backend": args.backend, "elapsed_s": elapsed,
        "config": vars(args),
        "headline": {
            "metamorphic_stability": results.get("metamorphic", {}).get("overall_metamorphic_stability"),
            "perturbation_degradation": results.get("perturbation", {}).get("degradation_clean_to_max_eps"),
            "fault_graceful": results.get("fault_injection", {}).get("graceful"),
            "adversarial_ASR": results.get("adversarial", {}).get("mean_attack_success_rate"),
            "passk_cls": results.get("passk", {}).get("classification", {}).get("pass_k_empirical"),
            "pass1_cls": results.get("passk", {}).get("classification", {}).get("pass_1_mean_acc"),
        },
    }
    json.dump(summary, open(f"outputs/live_stress_summary_{tag}.json", "w"), indent=2)
    print(f"\n[live] === DONE in {elapsed}s ===")
    print(json.dumps(summary["headline"], indent=2))
    print(f"[live] wrote outputs/live_stress_summary_{tag}.json")


if __name__ == "__main__":
    main()
