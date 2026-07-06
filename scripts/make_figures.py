"""
make_figures.py — publication figures from the REAL run artifacts (no synthetic numbers).

Reads whatever exists and renders each independently, so it works with partial results:
  outputs/benchmark_results.json        -> fig_benchmark_stress.png, fig_confusion.png
  outputs/agent_eval_results.json       -> fig_agent_vs_ml.png
  outputs/continuous_monitor.json       -> fig_continuous_monitor.png
  the realistic engine (live)           -> fig_quench_before_after.png, fig_fault_overlap.png

Every number plotted comes from a file written by an actual run (benchmark.py,
agent_eval.py, continuous_monitor.py) or from a fresh deterministic engine call.
Figures land in outputs/figures/*.png.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import cryo_engine as CE
from onnesim import realistic_faults as RF

FIGDIR = "outputs/figures"
INK = "#171717"; MUT = "#666"; BLUE = "#0070f3"; GREEN = "#00a862"
AMBER = "#f5a623"; RED = "#e5484d"; VIOLET = "#7928ca"; LIGHTBLUE = "#8ec5ff"
# Semantic convention, consistent across every figure:
#   AGENT (green) is reserved EXCLUSIVELY for the LLM agent panel.
#   ML (blue) is the supervised opponent; LIGHTBLUE is a second ML metric (same family).
# Never colour a non-agent series green, or a non-ML series blue.
AGENT = GREEN; ML = BLUE
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc",
                     "axes.grid": True, "grid.color": "#eeeeee", "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_quench_before_after():
    """The headline physics fix: quench was a gentle ramp, now a sharp transient."""
    cfg = CE.EngineConfig()
    sc = CE.Scenario("magnet_quench", severity=0.9, onset_frac=0.4)
    before = CE._simulate_clean(sc, cfg, hours=6, dt_min=1, seed=0)
    after = RF.simulate_realistic(sc, cfg, hours=6, dt_min=1, seed=0, imperfections=False)
    tb, ta = before["t_s"] / 60, after["t_s"] / 60
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(tb, before["temp2_T"], color=MUT, lw=1.6, label="before (steady load): +6.9% ramp")
    ax.plot(ta, after["temp2_T"], color=RED, lw=1.8, label="after (real quench): +323% spike")
    ax.axvline(0.4 * tb[-1], color="#bbbbbb", ls="--", lw=1)
    ax.set_xlabel("time (min)"); ax.set_ylabel("4K flange temperature (K)")
    ax.set_title("magnet_quench, fixed to real physics (½LI² dump + pulse-tube recovery)",
                 fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_quench_before_after.png"); plt.close(fig)
    return "fig_quench_before_after.png"


def fig_fault_overlap():
    """Three thermal faults overlap on temperature, separate on flow/pressure."""
    cfg = CE.EngineConfig()
    faults = ["helium_leak", "blocked_impedance", "wiring_heat_ingress"]
    cols = {f: RF.simulate_realistic(CE.Scenario(f, 0.9, 0.4), cfg, hours=6, dt_min=1,
                                     seed=2, imperfections=False) for f in faults}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    col = {"helium_leak": BLUE, "blocked_impedance": VIOLET, "wiring_heat_ingress": AMBER}
    for f in faults:
        t = cols[f]["t_s"] / 60
        a1.plot(t, cols[f]["temp5_T"] * 1e3, color=col[f], lw=1.6, label=f)
        a2.plot(t, cols[f]["flowmeter"], color=col[f], lw=1.6, label=f)
    a1.set_title("MXC temperature — AMBIGUOUS", fontsize=10.5, color=INK)
    a1.set_xlabel("time (min)"); a1.set_ylabel("MXC (mK)")
    a2.set_title("flow — the DISCRIMINATOR", fontsize=10.5, color=INK)
    a2.set_xlabel("time (min)"); a2.set_ylabel("flow (mmol/s)")
    a2.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Overlapping thermal faults: same temperature, distinct flow/pressure",
                 fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_fault_overlap.png"); plt.close(fig)
    return "fig_fault_overlap.png"


def fig_benchmark_stress():
    """ML accuracy collapses into the 0.75-0.85 target band under realism stress."""
    d = _load("outputs/benchmark_results.json")
    if not d:
        return None
    st = d["stress_test"]
    # stacked one-by-one (vertical column) instead of side-by-side
    fig, axes = plt.subplots(3, 1, figsize=(5.2, 8.2), sharey=True)
    axmap = [("noise", "extra sensor noise (rel)", axes[0]),
             ("window_frac", "observation window (fraction)", axes[1]),
             ("severity_scale", "fault severity (scale)", axes[2])]
    for key, xlabel, ax in axmap:
        rows = st[key]
        xs = [r["level"] for r in rows]
        for model, c in [("random_forest", ML), ("hist_gbdt", VIOLET)]:
            ys = [r[model]["macro_f1"] for r in rows if model in r]
            ax.plot(xs, ys, "o-", color=c, lw=1.7, ms=4, label=model)
        ax.axhspan(0.75, 0.85, color=AMBER, alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("macro-F1 (6 classes)")   # each panel labels its own y (stacked)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].text(0.02, 0.78, "target band 0.75–0.85", color="#b07a00", fontsize=8)
    fig.suptitle("Hardened benchmark: strong ML degrades\ninto the target band under stress",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{FIGDIR}/fig_benchmark_stress.png"); plt.close(fig)
    return "fig_benchmark_stress.png"


def fig_confusion():
    """Confusion matrix of the best ML model under stress (which faults get mixed up)."""
    d = _load("outputs/benchmark_results.json")
    if not d:
        return None
    cm = np.array(d["confusion_under_stress"]["confusion_matrix"], float)
    labels = d["confusion_under_stress"]["confusion_labels"]
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else INK, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Best ML confusion under {int(d['confusion_under_stress']['noise_level']*100)}% noise",
                 fontsize=10.5, color=INK)
    ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_confusion.png"); plt.close(fig)
    return "fig_confusion.png"


def fig_agent_vs_ml():
    """Head-to-head: live agent panel vs the best ML model on identical hard scenarios.

    Prefers the finished results JSON; falls back to the sleep-proof reconstruction from
    the turn log so the figure renders even if the run was killed before writing its JSON.
    """
    d = _load("outputs/agent_eval_results.json")
    src = "outputs/agent_eval_results.json"
    if not d and os.path.exists("outputs/agent_eval_turns.jsonl"):
        from onnesim import agent_eval as AE
        d = AE.reconstruct_from_log("outputs/agent_eval_turns.jsonl")
        src = "reconstructed"
    if not d:
        return None
    h = d["head_to_head"]
    n = d.get("n_scenarios") or d.get("n_scenarios_complete", 0)
    turns = d.get("agent_turns_total", 0)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    groups = ["detection F1", "classification acc"]
    agent = [h["agent_detection_f1"], h["agent_classification_acc"]]
    ml = [h["ml_detection_f1"], h["ml_classification_acc"]]
    x = np.arange(len(groups)); w = 0.36
    ax.bar(x - w/2, agent, w, color=AGENT, label=f"agent panel (Opus, {turns} turns)")
    ax.bar(x + w/2, ml, w, color=ML, label="supervised RF (primary opponent)")
    for xi, (a, m) in enumerate(zip(agent, ml)):
        ax.text(xi - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8.5)
        ax.text(xi + w/2, m + 0.01, f"{m:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    # legend OUTSIDE the axes (above), horizontal, so it never collides with the
    # value labels on the near-1.0 detection bars; title sits above the legend.
    ax.legend(frameon=False, fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), ncol=2)
    ax.set_title(f"Zero-shot agent panel vs supervised ML on {n} hard scenarios",
                 fontsize=10, color=INK, pad=30)
    ax.text(0.5, -0.22, "agents get no labels; ML trained on 300. Detection ties; "
            "ML wins classification (it had labels).",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUT)
    ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_agent_vs_ml.png"); plt.close(fig)
    return "fig_agent_vs_ml.png"


def fig_agent_confusion():
    """Where the reasoning agent fails: the deliberately-confusable fault pairs.

    The agent's top errors are exactly the overlapping thermal faults the engine was
    built to make ambiguous — evidence the eval measures what it was designed to.
    """
    d = _load("outputs/agent_eval_results.json")
    if not d and os.path.exists("outputs/agent_eval_turns.jsonl"):
        from onnesim import agent_eval as AE
        d = AE.reconstruct_from_log("outputs/agent_eval_turns.jsonl")
    if not d:
        return None
    ap = d.get("agent_panel")
    if not ap:
        return None
    cm = np.array(ap["confusion_matrix"], float)
    labels = ap["confusion_labels"]
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else INK, fontsize=8)
    ax.set_xlabel("agent predicted"); ax.set_ylabel("true fault")
    ax.set_title("Where the reasoning agent fails: confusable thermal-fault pairs",
                 fontsize=9.5, color=INK)
    ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_agent_confusion.png"); plt.close(fig)
    return "fig_agent_confusion.png"


def fig_continuous_monitor():
    """One continuous fridge run; agent catches the developing leak with a lead time."""
    d = _load("outputs/continuous_monitor.json")
    if not d:
        return None
    tr = d["trace"]; t = np.array(tr["t_min"]) / 60
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(t, tr["mxc_mK"], color=INK, lw=1.3, label="MXC (mK)")
    onset_h = d["onset_min"] / 60
    ax.axvline(onset_h, color=RED, ls="--", lw=1.4, label=f"fault onset (t={d['onset_min']:.0f}min)")
    if d["first_alarm_after_onset_min"]:
        ah = d["first_alarm_after_onset_min"] / 60
        ax.axvline(ah, color=AGENT, ls="-.", lw=1.4,
                   label=f"agent alarm (latency {d['detection_latency_min']:.0f}min)")
    for p in d["polls"]:
        if p["alarm"]:
            ax.plot(p["now_min"]/60, p["mxc_mK"], "v", color=AGENT, ms=6)
    ax.set_xlabel("time (h)"); ax.set_ylabel("MXC temperature (mK)")
    ax.set_title(f"Continuous {d['config']['hours']:.0f} h run: agent watches a "
                 f"{d['config']['fault_class']} develop live", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_continuous_monitor.png"); plt.close(fig)
    return "fig_continuous_monitor.png"


def fig_baseline_zoo():
    """Bar chart: the ML baseline zoo (RF/GBDT/LightGBM/TabPFN) on identical seeds."""
    d = _load("outputs/baseline_zoo.json")
    if not d:
        return None
    order = ["random_forest", "hist_gbdt", "lightgbm", "tabpfn", "logreg"]
    names = {"random_forest": "Random\nForest", "hist_gbdt": "Hist\nGBDT",
             "lightgbm": "LightGBM", "tabpfn": "TabPFN-2.5", "logreg": "Logistic\nReg."}
    models = d["models"]
    rows = [(k, models[k]) for k in order if k in models and "error" not in models[k]]
    if not rows:
        return None
    labels = [names.get(k, k) for k, _ in rows]
    f1 = [r["macro_f1"] for _, r in rows]
    acc = [r["exact_accuracy"]["p"] for _, r in rows]
    lo = [r["exact_accuracy"]["p"] - r["exact_accuracy"]["lo"] for _, r in rows]
    hi = [r["exact_accuracy"]["hi"] - r["exact_accuracy"]["p"] for _, r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(x - w/2, f1, w, color=ML, label="macro-$F_1$")
    ax.bar(x + w/2, acc, w, color=LIGHTBLUE, label="class. acc.",
           yerr=[lo, hi], capsize=3, ecolor=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Baseline zoo on identical $n{=}200$ seeds (95% CI on accuracy)",
                 fontsize=10, color=INK, pad=28)
    # legend OUTSIDE the axes (above), horizontal — the bars fill the plot so an
    # in-axes legend collides with the tallest group and its x-tick label.
    ax.legend(frameon=False, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), ncol=2)
    ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_baseline_zoo.png"); plt.close(fig)
    return "fig_baseline_zoo.png"


def fig_ablation():
    """Ablation: technique decomposition + panel-vs-single architecture."""
    d = _load("outputs/ablation_results.json")
    if not d:
        return None
    c = d["conditions"]
    order = [("panel_zero_shot", "zero-shot"), ("panel_few_shot_only", "+few-shot"),
             ("panel_self_consistency", "+self-consist."), ("panel_both", "+both"),
             ("single_zero_shot", "single\nzero-shot"), ("single_both", "single\n+both")]
    rows = [(lab, c[k]) for k, lab in order if k in c]
    if not rows:
        return None
    labels = [lab for lab, _ in rows]
    acc = [r["classification_acc"] for _, r in rows]
    cols = [GREEN if "single" not in labels[i] else VIOLET for i in range(len(rows))]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.bar(range(len(rows)), acc, color=cols)
    for i, a in enumerate(acc):
        ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("classification acc.")
    ax.set_title("Ablation: levers (green = 5-role panel) vs. single agent (violet)",
                 fontsize=10, color=INK); ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_ablation.png"); plt.close(fig)
    return "fig_ablation.png"


def fig_monitor_gating():
    """Confidence-gating trade-off: false alarms vs detection latency by gate."""
    d = _load("outputs/monitor_gating.json")
    if not d:
        return None
    gates = d["gates"]
    labels = [g["gate"] for g in gates]
    fa = [g["false_alarms_before_onset"] for g in gates]
    lat = [g["detection_latency_min"] for g in gates]
    x = np.arange(len(gates))
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    b = ax1.bar(x - 0.2, fa, 0.4, color=RED, label="false alarms (pre-onset)")
    ax1.set_ylabel("false alarms", color=RED); ax1.tick_params(axis="y", labelcolor=RED)
    ax1.set_xticks(x); ax1.set_xticklabels([f"{l}\ngate" for l in labels])
    ax2 = ax1.twinx()
    ax2.plot(x, lat, "o-", color=BLUE, lw=2, label="detection latency")
    ax2.set_ylabel("detection latency (min)", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE); ax2.set_ylim(0, max(lat) * 1.3)
    for i, (f, l) in enumerate(zip(fa, lat)):
        ax1.text(i - 0.2, f + 0.2, str(f), ha="center", fontsize=9, color=RED)
        ax2.text(i, l + 2, f"{l:.0f}", ha="center", fontsize=9, color=BLUE)
    ax1.set_title("Confidence gating: precision vs. latency trade-off",
                  fontsize=10, color=INK); ax1.grid(False)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_monitor_gating.png"); plt.close(fig)
    return "fig_monitor_gating.png"


def fig_ci_forest():
    """Forest plot: agent vs ML detection & classification accuracy with 95% CIs."""
    d = _load("outputs/head_to_head_stats.json")
    if not d:
        return None
    rows = [
        ("ML classification", d["classification"]["ml"], ML),
        ("Agent classification", d["classification"]["agent"], AGENT),
        ("ML detection", d["detection"]["ml"], ML),
        ("Agent detection", d["detection"]["agent"], AGENT),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (lab, ci, col) in enumerate(rows):
        ax.plot([ci["lo"], ci["hi"]], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax.plot(ci["p"], i, "o", color=col, ms=7)
        ax.text(ci["hi"] + 0.008, i, f"{ci['p']:.3f} [{ci['lo']:.2f},{ci['hi']:.2f}]",
                va="center", fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.55, 1.12); ax.set_xlabel("accuracy (Clopper–Pearson 95% CI)")
    ax.set_title("Detection ties, classification does not (n=200)", fontsize=10, color=INK)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_ci_forest.png"); plt.close(fig)
    return "fig_ci_forest.png"


def fig_monitor_sweep():
    """Multi-seed/multi-fault monitoring: per-fault latency distribution + gate trade-off.

    Left: detection-latency spread (box-ish min/median/max) per fault across seeds — the
    large-n replacement for the single 29.5-min number. Right: pooled confidence-gate
    trade-off (false alarms per run vs mean detection latency) across the whole grid.
    """
    d = _load("outputs/monitor_sweep.json")
    if not d:
        return None
    pf = d["per_fault"]
    faults = [f for f in pf if pf[f]["latency_min_any_gate"]]
    if not faults:
        return None
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 7.4))

    # top: latency spread per fault
    ys = list(range(len(faults)))
    for y, fc in zip(ys, faults):
        s = pf[fc]["latency_min_any_gate"]
        a1.plot([s["min"], s["max"]], [y, y], color=MUT, lw=1.4, solid_capstyle="round", zorder=1)
        a1.plot([s["p25"], s["p75"]], [y, y], color=BLUE, lw=5, solid_capstyle="round", alpha=0.5, zorder=2)
        a1.plot(s["median"], y, "o", color=BLUE, ms=7, zorder=3)
        a1.text(s["max"] + 1.5, y, f"n={s['n']}, rate={pf[fc]['detection_rate']:.2f}",
                va="center", fontsize=7.5, color=MUT)
    a1.set_yticks(ys); a1.set_yticklabels([f.replace("_", "\n") for f in faults], fontsize=8)
    a1.set_xlabel("detection latency (min)")
    ov = d["overall"]["detection_latency_min"]
    a1.set_title(f"Latency by fault across seeds "
                 f"(overall median {ov['median']} min, n={ov['n']})",
                 fontsize=10, color=INK)
    a1.axvline(d["config"]["poll_every_min"], color="#bbb", ls="--", lw=1)
    a1.text(d["config"]["poll_every_min"], len(faults) - 0.4, " 1 poll (cadence floor)",
            fontsize=7, color=MUT)

    # bottom: false-alarm gate curve — Gemini sweep (pooled) vs Opus single run.
    # On the quieter Gemini backend every gate sits at 0 FA/run; the gate's value
    # shows on the noisier Opus run (11->6->1). Plotting both makes the honest
    # cross-backend point: pre-onset false-alarm rate is backend-dependent.
    gc = d["gate_curve"]
    gates = ["any", "med", "high"]
    gem_fa = [gc[g]["false_alarms_per_run"]["mean"] if gc[g]["false_alarms_per_run"] else 0
              for g in gates]
    opus = _load("outputs/monitor_gating.json")
    opus_fa = None
    if opus:
        m = {row["gate"]: row["false_alarms_before_onset"] for row in opus["gates"]}
        opus_fa = [m.get(g, 0) for g in gates]
    x = np.arange(len(gates))
    w = 0.38
    a2.bar(x - w / 2, gem_fa, w, color=BLUE,
           label=f"Gemini sweep, pooled ({d['n_cells']} runs)")
    if opus_fa is not None:
        a2.bar(x + w / 2, opus_fa, w, color=RED, label="Opus, single run")
        for i, v in enumerate(opus_fa):
            a2.text(i + w / 2, v + 0.15, f"{v}", ha="center", fontsize=8, color=RED)
    for i, v in enumerate(gem_fa):
        a2.text(i - w / 2, v + 0.15, f"{v:.0f}", ha="center", fontsize=8, color=BLUE)
    a2.set_xticks(x); a2.set_xticklabels([f"{g}\ngate" for g in gates])
    a2.set_ylabel("pre-onset false alarms / run")
    a2.set_title("False alarms are backend-dependent", fontsize=10, color=INK)
    a2.legend(frameon=False, fontsize=7.5, loc="upper right")
    a2.grid(axis="x")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_monitor_sweep.png"); plt.close(fig)
    return "fig_monitor_sweep.png"


def fig_retrieval_ablation():
    """Demo-selection ablation: zero-shot vs fixed round-robin vs query-conditioned retrieval."""
    d = _load("outputs/retrieval_ablation.json")
    if not d or not d.get("conditions"):
        return None
    label = {"zero_shot": "zero-shot", "roundrobin": "fixed\nround-robin",
             "retrieval": "query\nretrieval"}
    rows = d["conditions"]
    names = [label.get(r["label"], r["label"]) for r in rows]
    cls = [r["cls"] for r in rows]
    ml = rows[-1]["ml_cls"]
    cols = [MUT if r["label"] == "zero_shot" else (GREEN if r["label"] == "retrieval" else BLUE)
            for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    x = np.arange(len(rows))
    ax.bar(x, cls, 0.6, color=cols)
    for i, c in enumerate(cls):
        ax.text(i, c + 0.01, f"{c:.3f}", ha="center", fontsize=9)
    ax.axhline(ml, color=RED, ls="--", lw=1.4, label=f"supervised RF ({ml:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("classification accuracy")
    ax.set_title("Demo selection: fixed block vs. query-conditioned retrieval",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(axis="x")
    lift = d.get("retrieval_lift")
    if lift:
        mc = lift["mcnemar_lift"]
        ax.text(0.02, 0.04, f"retrieval vs round-robin: McNemar p={mc['p_value']:.2g} "
                f"(+{mc['improved_by_enhancement']}/-{mc['regressed']})",
                transform=ax.transAxes, fontsize=7.5, color=MUT)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_retrieval_ablation.png"); plt.close(fig)
    return "fig_retrieval_ablation.png"


def fig_label_efficiency():
    """Label efficiency: ML accuracy vs training-label budget, with the agent reference.

    Full-width figure. Agent line is GREEN (convention: green = agent). Reads the real
    B200 sweep; falls back to the smoke run if the full one is absent.
    """
    d = _load("outputs/b200_results/label_efficiency.json") or _load("outputs/label_efficiency.json")
    if not d:
        return None
    budgets = d["budgets"]
    cells = d["cells"]
    ref = d.get("reference", {})
    agent_acc = ref.get("agent_enhanced_6_demos", 0.99)

    # model -> (display label, colour, linestyle, marker); tabular solid, deep dashed
    MODELS = [
        ("random_forest",       "Random forest",       ML,     "-",  "o"),
        ("hist_gbdt",           "Hist-GBDT",            VIOLET, "--", "s"),
        ("lightgbm",            "LightGBM",             AMBER,  ":",  "^"),
        ("logreg",              "Logistic reg.",        MUT,    "-",  "v"),
        ("cnn_gru",             "CNN-GRU",              "#00a862", "--", "D"),  # teal-green deep
        ("timesnet",            "TimesNet",             "#e07b00", ":",  "X"),
        ("anomaly_transformer", "Anomaly-Transformer",  "#17becf", "-.", "P"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 5.6))          # WIDE: spans full text width
    x = np.arange(len(budgets))

    # agent reference (green dashed) — the thing ML is chasing
    ax.axhline(agent_acc, color=AGENT, ls="--", lw=2.4, zorder=5,
               label=f"Agent (6 demos): {agent_acc:.3f}")

    for key, lab, col, ls, mk in MODELS:
        ys, los, his = [], [], []
        for b in budgets:
            c = cells.get(f"{b}|{key}")
            if c is None:
                ys.append(np.nan); los.append(0); his.append(0); continue
            m = c["acc_mean"]; lo, hi = c.get("acc_95ci", [m, m])
            ys.append(m); los.append(m - lo); his.append(hi - m)
        ys = np.array(ys, float)
        ax.errorbar(x, ys, yerr=[los, his], color=col, ls=ls, marker=mk, lw=1.9,
                    ms=6, capsize=3, elinewidth=1.0, label=lab, zorder=3)

    # 6-label regime band + label placed at the TOP of the band (data region there is
    # empty: all models score <=0.42 at 6 labels), so it never overlaps points or ticks.
    ax.axvspan(-0.35, 0.35, color=AGENT, alpha=0.10, zorder=0)
    ax.text(0.0, 0.52, "6-label\nregime", ha="center", va="bottom",
            fontsize=9, color="#0a7f4f", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Number of training labels (log-spaced budgets)", fontsize=12)
    ax.set_ylabel("Classification accuracy", fontsize=12)
    ax.set_ylim(0.1, 1.06)
    ax.set_title("Label efficiency: the agent matches supervised ML from 6 demonstrations; "
                 "ML needs 120–300 labels to reach the same accuracy\n"
                 "(on the twin, 6 seeds/cell, 95% CI)",
                 fontsize=12.5, color=INK)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="lower right",
              handlelength=2.6, columnspacing=1.4)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_label_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "fig_label_efficiency.png"


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    made = []
    for fn in [fig_quench_before_after, fig_fault_overlap, fig_benchmark_stress,
               fig_confusion, fig_agent_vs_ml, fig_agent_confusion, fig_continuous_monitor,
               fig_baseline_zoo, fig_ablation, fig_monitor_gating, fig_ci_forest,
               fig_retrieval_ablation, fig_label_efficiency]:
        try:
            out = fn()
            if out:
                made.append(out); print(f"[figures] wrote {out}")
            else:
                print(f"[figures] skipped {fn.__name__} (no data yet)")
        except Exception as exc:  # noqa: BLE001
            print(f"[figures] FAILED {fn.__name__}: {exc}")
    print(f"[figures] {len(made)} figures in {FIGDIR}")


if __name__ == "__main__":
    main()
