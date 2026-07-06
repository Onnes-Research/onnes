#!/usr/bin/env python3
"""
analyze_turn_logs.py — recompute paper numbers that live ONLY in the released
per-turn logs (not in the summary JSONs), so they are auditable the same way
onnesim/stats.py recomputes the CIs/McNemar.

Produces three families of numbers, all VERBATIM from the turn logs:

  1. Guardian veto rate  — how often the Guardian role returned approved=False,
     across the 1000-turn zero-shot run and the enhanced run. Substantiates the
     paper's architectural claim ("the panel buys an auditable, separable veto
     surface") with an actual number instead of an assertion.

  2. Per-class precision / recall / F1 / support — for the agent (zero-shot),
     the enhanced panel, and the shared RF opponent, on the SAME n=200 scenarios.
     Quantifies "errors concentrate on the engineered confusable pairs".

  3. Confusion matrices with ABSOLUTE COUNTS for zero-shot, enhanced, and RF,
     so the confusion figure can be re-rendered with counts and the enhanced/RF
     panels added.

Everything is reconstructed from the logged (truth, agent_pred, ml_pred, role,
reply) tuples — the same fields onnesim/agent_eval.reconstruct_from_log uses.

Usage:
    python scripts/analyze_turn_logs.py
    # writes outputs/turn_log_analysis.json and prints a summary
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

# Canonical class order (matches onnesim.cryo_engine.FAULT_CLASSES).
CLASSES = ["normal", "heat_load_spike", "helium_leak",
           "magnet_quench", "wiring_heat_ingress", "blocked_impedance"]

# The two headline runs and their logs (both live LLM = Claude/Opus via litellm).
RUNS = {
    "zero_shot": "outputs/agent_eval_turns.jsonl",
    "enhanced":  "outputs/agent_eval_fewshot_turns.jsonl",
}


def _per_scenario(log_path: str) -> dict[int, dict]:
    """Collapse a turn log to one record per scenario: truth, agent_pred, ml_pred,
    and the Guardian's approved flag (None if the Guardian JSON did not parse)."""
    scen: dict[int, dict] = defaultdict(lambda: {"guardian_approved": None})
    for line in open(log_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = scen[r["scenario"]]
        s["truth"] = r.get("truth")
        s["agent_pred"] = r.get("agent_pred")
        s["ml_pred"] = r.get("ml_pred")
        if r.get("role") == "guardian":
            rep = r.get("reply")
            s["guardian_approved"] = (bool(rep["approved"])
                                      if isinstance(rep, dict) and "approved" in rep
                                      else None)
    return scen


def guardian_veto_stats(scen: dict[int, dict]) -> dict:
    """Veto rate over scenarios whose Guardian turn returned a parseable approved flag."""
    tally = Counter()
    for s in scen.values():
        a = s["guardian_approved"]
        tally["parsed" if a is not None else "unparsed"] += 1
        if a is False:
            tally["vetoed"] += 1
        elif a is True:
            tally["approved"] += 1
    parsed = tally["parsed"]
    return {
        "n_scenarios": len(scen),
        "guardian_parsed": parsed,
        "guardian_unparsed": tally["unparsed"],
        "approved": tally["approved"],
        "vetoed": tally["vetoed"],
        "veto_rate_of_parsed": round(tally["vetoed"] / parsed, 4) if parsed else None,
        "veto_rate_of_all": round(tally["vetoed"] / len(scen), 4) if scen else None,
    }


def confusion_counts(scen: dict[int, dict], pred_key: str) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for s in scen.values():
        t, p = s.get("truth"), s.get(pred_key)
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def per_class_metrics(cm: list[list[int]]) -> dict:
    """Precision/recall/F1/support per class from a confusion-count matrix, plus macro-F1."""
    n = len(CLASSES)
    out = {}
    f1s = []
    for i, c in enumerate(CLASSES):
        tp = cm[i][i]
        support = sum(cm[i])                      # true instances of class c
        pred_pos = sum(cm[r][i] for r in range(n))  # predicted as class c
        prec = tp / pred_pos if pred_pos else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(f1, 3), "support": support}
        if support:
            f1s.append(f1)
    out["_macro_f1"] = round(sum(f1s) / len(f1s), 3) if f1s else 0.0
    out["_overall_acc"] = round(
        sum(cm[i][i] for i in range(n)) / max(sum(sum(r) for r in cm), 1), 3)
    return out


def main() -> None:
    result = {"classes": CLASSES, "runs": {}, "note": (
        "All numbers recomputed verbatim from the released per-turn JSONL logs. "
        "Guardian veto = fraction of scenarios whose Guardian role returned "
        "approved=False. Per-class metrics and confusion counts are over the "
        "identical n=200 held-out scenarios shared by agent and RF.")}

    for name, path in RUNS.items():
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        scen = _per_scenario(path)
        cm_agent = confusion_counts(scen, "agent_pred")
        cm_ml = confusion_counts(scen, "ml_pred")
        result["runs"][name] = {
            "log_path": path,
            "guardian": guardian_veto_stats(scen),
            "agent_confusion_counts": cm_agent,
            "agent_per_class": per_class_metrics(cm_agent),
            "ml_confusion_counts": cm_ml,
            "ml_per_class": per_class_metrics(cm_ml),
        }

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/turn_log_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # ---- human-readable summary --------------------------------------------
    print(f"wrote {out_path}\n")
    for name, r in result["runs"].items():
        g = r["guardian"]
        print(f"== {name} ({r['log_path']}) ==")
        print(f"  Guardian: vetoed {g['vetoed']}/{g['guardian_parsed']} parsed "
              f"({g['veto_rate_of_parsed']:.1%} of parsed; "
              f"{g['vetoed']}/{g['n_scenarios']} = {g['veto_rate_of_all']:.1%} of all; "
              f"{g['guardian_unparsed']} unparsed)")
        ap = r["agent_per_class"]
        print(f"  Agent macro-F1={ap['_macro_f1']}  acc={ap['_overall_acc']}")
        for c in CLASSES:
            m = ap[c]
            print(f"    {c:20s} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} n={m['support']}")
        print()


if __name__ == "__main__":
    main()
