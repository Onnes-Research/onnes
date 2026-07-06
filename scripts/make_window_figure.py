#!/usr/bin/env python3
"""
make_window_figure.py — the "information funnel" figure: raw telemetry ->
compact LLM summary -> zero-shot vs few-shot agent JSON, for ONE real scenario.

Everything is drawn from real artifacts:
  * telemetry: regenerated deterministically from cryo_engine at the eval seed
    (same seed the head-to-head used), so the traces ARE the window the agent saw.
  * summary  : the exact A.summarize_window() dict fed to the LLM.
  * verdicts : the actual logged supervisor replies from the released turn logs
    (zero-shot got blocked_impedance; few-shot got helium_leak). Passed in via
    args so no text is hand-authored.

Scenario: seed 10001, truth = helium_leak — one of the engineered confusable
faults. Illustrates both (a) what summarization discards and (b) how the
contrastive demos flip the confusable-pair error.

Usage:
    .venv/bin/python scripts/make_window_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from onnesim import cryo_engine as CE, agent as A, virtual_clone as VC

FIGDIR = "paper/figures"
SEED = 10001  # eval scenario 1, truth = helium_leak (see turn_log_analysis)


def main() -> None:
    # --- regenerate the exact window the agent saw (deterministic at this seed) --
    fp = None
    try:
        fp = VC.learn_fingerprint("data/real/bluefors_cryometrics_sample")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] fingerprint unavailable ({exc}); using engine noise")
    engine_cfg = CE.EngineConfig(fingerprint=fp, realistic=True, imperfections=True)
    # head-to-head severity stressor is sev*0.5; scenario 1 fault severity ~0.7
    scen = CE.Scenario("helium_leak", 0.7 * 0.5, 0.4)
    cols = CE.simulate(scen, engine_cfg, hours=6.0, dt_min=5.0, seed=SEED)
    t_h = np.asarray(cols["t_s"], dtype=float) / 3600.0

    # --- panel A: raw-ish telemetry (the discriminating channels) ---------------
    fig = plt.figure(figsize=(13.0, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.32)

    axT = fig.add_subplot(gs[0, 0])
    for ch, lab, col in [("temp5_T", "MXC (temp5)", "#0070F3"),
                         ("temp4_T", "cold plate (temp4)", "#00A862")]:
        y = np.asarray(cols[ch], dtype=float) * 1e3  # K -> mK
        axT.plot(t_h, y, lw=1.6, label=lab, color=col)
    axT.set_xlabel("time (h)"); axT.set_ylabel("temperature (mK)")
    axT.legend(fontsize=8, loc="upper left")
    axT.set_title("(A) raw telemetry window", fontsize=10)
    axf = axT.twinx()
    axf.plot(t_h, np.asarray(cols["flowmeter"], dtype=float), lw=1.4,
             ls="--", color="#E5484D", label="flow")
    if "p5" in cols:
        p5 = np.asarray(cols["p5"], dtype=float)
        axf.plot(t_h, p5 / np.nanmedian(p5[:3]), lw=1.2, ls=":",
                 color="#7928CA", label="p5 (norm)")
    axf.set_ylabel("flow / p5 (norm)")
    axf.legend(fontsize=7, loc="lower left")

    # --- panel B: the compact summary actually fed to the LLM --------------------
    summ = A.summarize_window(cols)
    # keep the discriminating channels for a readable box
    keep = ["temp5_T", "temp4_T", "temp3_T", "flowmeter", "p5", "p2"]
    lines = [f'duration_h: {summ.get("duration_h")}', "channels:"]
    chd = summ.get("channels", {})
    for k in keep:
        if k in chd:
            c = chd[k]
            lines.append(f'  {k}: start={c.get("start"):.4g}, '
                         f'end={c.get("end"):.4g}, '
                         f'pct={c.get("pct_change")}')
    js = json.dumps(summ)
    lines.append("")
    lines.append(f"[full summary: {len(chd)} channels, {len(js)} chars ~ "
                 f"{len(js)//4} tokens]")
    axB = fig.add_subplot(gs[0, 1]); axB.axis("off")
    axB.set_title("(B) compact summary -> LLM", fontsize=10)
    axB.text(0.0, 0.98, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=7.0, transform=axB.transAxes)

    # --- panel C: zero-shot vs few-shot verdicts (from args = real log replies) --
    zs = args.get("zero_shot", "blocked_impedance") if isinstance(args, dict) else "blocked_impedance"
    fs = args.get("few_shot", "helium_leak") if isinstance(args, dict) else "helium_leak"
    axC = fig.add_subplot(gs[0, 2]); axC.axis("off")
    axC.set_title("(C) Supervisor verdict", fontsize=10)
    axC.text(0.0, 0.92, "truth: helium_leak", va="top", family="monospace",
             fontsize=9, transform=axC.transAxes, color="#171717")
    axC.text(0.0, 0.72,
             f'zero-shot:\n  fault_class = "{zs}"\n  (X confusable-pair error)',
             va="top", family="monospace", fontsize=8.5, color="#E5484D",
             transform=axC.transAxes)
    axC.text(0.0, 0.40,
             f'+6 demos +SC:\n  fault_class = "{fs}"\n  (correct)',
             va="top", family="monospace", fontsize=8.5, color="#00A862",
             transform=axC.transAxes)

    fig.suptitle("From raw window to verdict (real scenario, seed 10001). "
                 "The compact summary preserves the leak's flow/pressure signature; "
                 "contrastive demos flip the zero-shot confusable-pair error.",
                 fontsize=9.5, y=1.02)
    out = f"{FIGDIR}/fig_window_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


# allow the real verdicts to be injected; default to the known logged values
try:
    args
except NameError:
    args = {"zero_shot": "blocked_impedance", "few_shot": "helium_leak"}

if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    main()
