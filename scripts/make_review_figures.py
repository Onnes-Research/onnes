#!/usr/bin/env python3
"""
make_review_figures.py — regenerate the figures added in the pre-submission
validity/presentation pass. Every figure is drawn from a released artifact so it
stays consistent with the "all numbers from logs" contract.

Outputs (paper/figures/):
  fig_confusion_panels.png   — zero-shot / enhanced / RF confusion matrices with
                               ABSOLUTE COUNTS, from outputs/turn_log_analysis.json
                               (run scripts/analyze_turn_logs.py first).
  fig_label_efficiency_bands.png — per-budget accuracy with shaded 95% CI bands
                               over the 6 seeds/cell + horizontal reference lines
                               for zero-shot (0.685) and enhanced 6-demo (0.990),
                               from outputs/label_efficiency.json.

Usage:
    python scripts/make_review_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = "paper/figures"
CLASS_SHORT = ["norm", "heat", "He-leak", "quench", "wiring", "block"]


# --------------------------------------------------------------------------- #
def confusion_panels() -> None:
    src = "outputs/turn_log_analysis.json"
    if not os.path.exists(src):
        print(f"[skip] {src} missing — run scripts/analyze_turn_logs.py first")
        return
    d = json.load(open(src))
    panels = [
        ("Zero-shot agent", d["runs"]["zero_shot"]["agent_confusion_counts"]),
        ("Enhanced agent (6 demos + SC)", d["runs"]["enhanced"]["agent_confusion_counts"]),
        ("Supervised RF", d["runs"]["zero_shot"]["ml_confusion_counts"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (title, cm) in zip(axes, panels):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CLASS_SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_SHORT, fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("true", fontsize=9)
        thr = cm.max() / 2 if cm.max() else 1
        for i in range(6):
            for j in range(6):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > thr else "#222", fontsize=9)
    fig.suptitle("Confusion matrices (absolute counts, n=200). Zero-shot errors "
                 "collapse onto the engineered confusable pairs; the enhanced panel "
                 "resolves them.", fontsize=10, y=1.02)
    fig.tight_layout()
    out = f"{FIGDIR}/fig_confusion_panels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
def label_efficiency_bands() -> None:
    src = "outputs/label_efficiency.json"
    if not os.path.exists(src):
        print(f"[skip] {src} missing")
        return
    d = json.load(open(src))
    cells = d["cells"]
    budgets = sorted({v["n_labels"] for v in cells.values()})
    models = sorted({v["model"] for v in cells.values()})

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for m in models:
        xs, mean, lo, hi = [], [], [], []
        for b in budgets:
            c = cells.get(f"{b}|{m}")
            if not c:
                continue
            xs.append(b)
            mean.append(c["acc_mean"])
            ci = c.get("acc_95ci", [c["acc_mean"], c["acc_mean"]])
            lo.append(ci[0]); hi.append(ci[1])
        if not xs:
            continue
        line, = ax.plot(xs, mean, marker="o", ms=3.5, lw=1.4, label=m)
        ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15)

    ref = d.get("reference", {})
    if "agent_enhanced_6_demos" in ref:
        ax.axhline(ref["agent_enhanced_6_demos"], ls="--", lw=1.6, color="#7a3fbf",
                   label=f"enhanced panel, 6 demos ({ref['agent_enhanced_6_demos']})")
    if "agent_zero_shot_0_labels" in ref:
        ax.axhline(ref["agent_zero_shot_0_labels"], ls=":", lw=1.4, color="#555",
                   label=f"zero-shot panel, 0 labels ({ref['agent_zero_shot_0_labels']})")

    ax.set_xscale("log")
    ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("labeled training scenarios (log scale)")
    ax.set_ylabel("multiclass accuracy")
    ax.set_title("Label efficiency: supervised models vs. training-free panel\n"
                 "(bands = 95% CI over 6 seeds/cell)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    out = f"{FIGDIR}/fig_label_efficiency_bands.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    confusion_panels()
    label_efficiency_bands()
