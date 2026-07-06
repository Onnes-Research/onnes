"""
finalize_paper.py — fill the paper's placeholder macros from the REAL run artifacts.

Run this after the enhanced n=200 and ablation jobs finish. It reads the JSON artifacts,
computes the paired-lift McNemar test, and rewrites the \newcommand macro block near the
top of paper/onnes.tex so the numbers in the tables/prose are a single source of truth.
Idempotent: it only rewrites the delimited macro block, leaving the rest of the tex alone.

Never invents numbers: if an artifact is missing, the corresponding macro keeps its
'[run in progress]'/'--' placeholder and the script says so.
"""
from __future__ import annotations
import json
import os
import re

TEX = "paper/onnes.tex"
BEGIN = "% >>> AUTO-FILLED MACROS >>>"
END = "% <<< AUTO-FILLED MACROS <<<"


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _acc_from_results(path):
    d = _load(path)
    if not d:
        return None
    h = d.get("head_to_head", {})
    return h.get("agent_classification_acc")


def build_macros() -> dict:
    macros = {}

    # enhanced n=200 classification accuracy
    enh = _acc_from_results("outputs/agent_eval_fewshot_n200_results.json")
    if enh is not None:
        macros["CONFTWOHUNDRED"] = f"${enh:.3f}$"
        macros["CONFTWOHUNDREDSHORT"] = f"{enh:.3f}"

    # ablation conditions
    abl = _load("outputs/ablation_results.json")
    if abl:
        c = abl["conditions"]
        def g(label):
            r = c.get(label)
            return f"${r['classification_acc']:.3f}$" if r else "--"
        macros["ABLpanelZero"] = g("panel_zero_shot")
        macros["ABLpanelFS"] = g("panel_few_shot_only")
        macros["ABLpanelSC"] = g("panel_self_consistency")
        macros["ABLpanelBoth"] = g("panel_both")
        macros["ABLsingleZero"] = g("single_zero_shot")
        macros["ABLsingleBoth"] = g("single_both")
    return macros


def paired_lift_stat():
    """Compute the paired McNemar lift zero-shot -> enhanced on shared n=200 seeds."""
    try:
        from onnesim import stats as ST
    except Exception:  # noqa: BLE001
        return None
    zero = "outputs/agent_eval_turns.jsonl"
    enh = "outputs/agent_eval_fewshot_turns.jsonl"
    if not (os.path.exists(zero) and os.path.exists(enh)):
        return None
    try:
        r = ST.paired_lift(zero, enh)
        json.dump(r, open("outputs/paired_lift_stats.json", "w"), indent=2)
        return r
    except Exception as exc:  # noqa: BLE001
        print(f"[finalize] paired_lift failed: {exc}")
        return None


DEFAULTS = {
    "CONFTWOHUNDRED": "[n200 pending]",
    "CONFTWOHUNDREDSHORT": "--",
    "ABLpanelZero": "--", "ABLpanelFS": "--", "ABLpanelSC": "--",
    "ABLpanelBoth": "--", "ABLsingleZero": "--", "ABLsingleBoth": "--",
}


def rewrite_tex(macros: dict):
    r"""Replace each \newcommand{\Name}{...} line for a known macro with the filled value."""
    with open(TEX) as f:
        tex = f.read()
    filled = {**DEFAULTS, **macros}
    for name, val in filled.items():
        pat = re.compile(r"\\newcommand\{\\" + name + r"\}\{[^}]*\}")
        repl = "\\newcommand{\\" + name + "}{" + val + "}"
        tex, n = pat.subn(lambda _m, r=repl: r, tex)  # lambda -> literal, no backref parsing
        if n == 0:
            print(f"[finalize] WARNING: macro {name} not found in {TEX}")
    with open(TEX, "w") as f:
        f.write(tex)


def main():
    macros = build_macros()
    print(f"[finalize] filled {len(macros)} macros from artifacts:")
    for k, v in macros.items():
        print(f"    {k} = {v}")
    lift = paired_lift_stat()
    if lift:
        mc = lift["mcnemar_lift"]
        print(f"[finalize] paired lift n={lift['n_shared_scenarios']}: "
              f"zero={lift['zero_shot']['p']:.3f} enhanced={lift['enhanced']['p']:.3f} "
              f"McNemar p={mc['p_value']:.2e} (improved {mc['improved_by_enhancement']}, "
              f"regressed {mc['regressed']})")
    rewrite_tex(macros)
    print(f"[finalize] rewrote macro block in {TEX}")


if __name__ == "__main__":
    main()
