"""
stats.py — statistical rigor for the head-to-head, computed from the REAL turn log.

Every reviewer (Gemini, ChatGPT, Grok, Claude) flagged the same gap: the paper
reports point estimates (F1, accuracy) with no uncertainty and no significance test.
This module closes that gap with exact, assumption-light methods:

  * Clopper-Pearson exact binomial confidence intervals on every proportion
    (detection accuracy, classification accuracy). Exact (not normal-approx), so
    it is valid at the small n of the held-out subset too.
  * Exact McNemar test on the PAIRED agent-vs-ML predictions (same scenarios), the
    correct test for "are these two classifiers different on the same items?".
    Also used for the zero-shot vs enhanced paired lift.

Nothing here calls a model or an RNG: it reads the existing JSONL logs, so it is
fully reproducible and adds no API cost. Numbers flow into the paper's tables.
"""
from __future__ import annotations
import json
from scipy import stats as _st

from . import agent_eval as AE

NORMAL = "normal"


# ----------------------------------------------------------- proportions ----
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 100(1-alpha)% CI for a binomial proportion k/n.

    Valid for any n including small held-out sets, and never leaves [0,1].
    Uses the Beta-quantile form: lo = Beta(alpha/2; k, n-k+1),
    hi = Beta(1-alpha/2; k+1, n-k)."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _st.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _st.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def prop_ci(k: int, n: int, alpha: float = 0.05) -> dict:
    lo, hi = clopper_pearson(k, n, alpha)
    return {"k": int(k), "n": int(n), "p": (k / n if n else 0.0),
            "lo": round(lo, 3), "hi": round(hi, 3),
            "ci_pct": int(round((1 - alpha) * 100))}


# --------------------------------------------------------------- McNemar ----
def mcnemar_exact(b: int, c: int) -> dict:
    """Exact McNemar test on the two discordant counts of a paired 2x2 table.

    b = # items where classifier A is right and B is wrong,
    c = # items where A is wrong and B is right.
    Under H0 (equal error rates) each discordant item is a fair coin, so the exact
    p-value is the two-sided binomial tail of min(b,c) out of b+c at p=0.5.
    Concordant cells are irrelevant (they cancel), which is the whole point."""
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "note": "no discordant pairs"}
    p = float(_st.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return {"b": int(b), "c": int(c), "n_discordant": n, "p_value": p}


def _paired_correct(y_true, y_a, y_b):
    """Discordant counts for McNemar on classification correctness of A vs B."""
    b = c = 0
    for t, a, bb in zip(y_true, y_a, y_b):
        ca, cb = (a == t), (bb == t)
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    return b, c


# ------------------------------------------------------- log -> statistics --
def _load_complete(log_path: str):
    """Return aligned (truth, agent_pred, ml_pred, detected) over COMPLETE scenarios,
    in scenario order, reusing agent_eval's own normalization so labels match tables."""
    per: dict[int, dict] = {}
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = per.setdefault(r["scenario"], {"roles": set(), "detected": None})
            s["truth"] = AE._norm_class(r["truth"])
            s["agent"] = AE._norm_class(r["agent_pred"])
            s["ml"] = AE._norm_class(r["ml_pred"])
            s["roles"].add(r["role"])
            if r["role"] == "supervisor" and isinstance(r.get("reply"), dict):
                s["detected"] = bool(r["reply"].get("fault_detected", s["agent"] != NORMAL))
    keys = sorted(k for k, s in per.items() if "supervisor" in s["roles"])
    truth = [per[k]["truth"] for k in keys]
    agent = [per[k]["agent"] for k in keys]
    ml = [per[k]["ml"] for k in keys]
    det = [(per[k]["detected"] if per[k]["detected"] is not None
            else per[k]["agent"] != NORMAL) for k in keys]
    return truth, agent, ml, det


def analyze_head_to_head(log_path: str = "outputs/agent_eval_turns.jsonl") -> dict:
    """CIs on agent/ML detection+classification accuracy, plus paired McNemar tests.

    Detection here is the binary normal-vs-fault decision; for the agent we use the
    Supervisor's explicit flag (same as the scorecard). Classification is the exact
    fault-class match on ALL scenarios (a wrong 'normal' counts as wrong)."""
    truth, agent, ml, det = _load_complete(log_path)
    n = len(truth)

    # detection correctness (binary) per method
    det_true = [t != NORMAL for t in truth]
    agent_det_correct = sum(int(d == dt) for d, dt in zip(det, det_true))
    ml_det_correct = sum(int((m != NORMAL) == dt) for m, dt in zip(ml, det_true))

    # classification correctness (exact class) per method
    agent_cls_correct = sum(int(a == t) for a, t in zip(agent, truth))
    ml_cls_correct = sum(int(m == t) for m, t in zip(ml, truth))

    # paired McNemar: agent vs ML on classification correctness
    b, c = _paired_correct(truth, agent, ml)
    mc_cls = mcnemar_exact(b, c)
    # paired McNemar: agent vs ML on detection correctness
    bd, cd = 0, 0
    for dt, d, m in zip(det_true, det, ml):
        ca, cb = (d == dt), ((m != NORMAL) == dt)
        if ca and not cb:
            bd += 1
        elif cb and not ca:
            cd += 1
    mc_det = mcnemar_exact(bd, cd)

    return {
        "log_path": log_path, "n_scenarios": n,
        "detection": {
            "agent": prop_ci(agent_det_correct, n),
            "ml": prop_ci(ml_det_correct, n),
            "mcnemar_agent_vs_ml": mc_det,
        },
        "classification": {
            "agent": prop_ci(agent_cls_correct, n),
            "ml": prop_ci(ml_cls_correct, n),
            "mcnemar_agent_vs_ml": mc_cls,
        },
        "reading": (
            "Clopper-Pearson 95% CIs; McNemar is the exact paired binomial on discordant "
            "pairs (same scenarios for both methods). A small McNemar p on classification "
            "means the ML advantage is not sampling noise; a large p on detection means "
            "'detection ties' is a real statistical statement, not just close point values."
        ),
    }


def paired_lift(zero_log: str, enhanced_log: str) -> dict:
    """McNemar paired test for the zero-shot -> enhanced classification lift on the
    scenarios present in BOTH logs (matched by scenario id). This is the correct test
    for the headline 'the two techniques improve classification' claim."""
    zt, za, _, _ = _load_complete(zero_log)
    et, ea, _, _ = _load_complete(enhanced_log)
    # re-key by scenario id for intersection
    def keyed(log):
        per = {}
        with open(log) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                s = per.setdefault(r["scenario"], {"roles": set()})
                s["truth"] = AE._norm_class(r["truth"])
                s["pred"] = AE._norm_class(r["agent_pred"])
                s["roles"].add(r["role"])
        return {k: v for k, v in per.items() if "supervisor" in v["roles"]}
    z, e = keyed(zero_log), keyed(enhanced_log)
    shared = sorted(set(z) & set(e))
    truth = [z[k]["truth"] for k in shared]
    zpred = [z[k]["pred"] for k in shared]
    epred = [e[k]["pred"] for k in shared]
    b, c = 0, 0  # b: enhanced right & zero wrong (the lift); c: zero right & enhanced wrong
    for t, zp, ep in zip(truth, zpred, epred):
        cz, ce = (zp == t), (ep == t)
        if ce and not cz:
            b += 1
        elif cz and not ce:
            c += 1
    mc = mcnemar_exact(b, c)
    nz = sum(int(zp == t) for zp, t in zip(zpred, truth))
    ne = sum(int(ep == t) for ep, t in zip(epred, truth))
    n = len(shared)
    return {
        "n_shared_scenarios": n,
        "zero_shot": prop_ci(nz, n),
        "enhanced": prop_ci(ne, n),
        "mcnemar_lift": {**mc, "improved_by_enhancement": b, "regressed": c},
        "reading": ("Paired McNemar on identical scenarios. b = scenarios the enhancement "
                    "fixed, c = scenarios it broke; a small p means the lift is significant."),
    }


if __name__ == "__main__":
    import sys
    out = analyze_head_to_head(sys.argv[1] if len(sys.argv) > 1
                               else "outputs/agent_eval_turns.jsonl")
    print(json.dumps(out, indent=2))
