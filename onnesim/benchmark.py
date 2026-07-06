"""
benchmark.py — the HONEST, HARDER CryoOpsBench harness.

Motivation (docs/FINDINGS.md #1): the fault classifier scores F1 0.997 on the
engine. Near-perfect on synthetic data is a RED FLAG, not a trophy — it means the
injected fault signatures are too cleanly separable and won't transfer to real
fridges. A *good* benchmark makes strong models land ~0.75-0.85, not 1.0. This
module builds the evaluation that measures where models actually stand and, just
as important, says so plainly when the engine is still too easy.

What it does (all from the UNCHANGED engine, imported as-is):
  1. HONEST protocol — generate scenarios from cryo_engine, hold out BY SEED
     (train seeds and test seeds are disjoint, so no noise realization leaks),
     with optional label-noise robustness sweeps.
  2. MULTI-MODEL comparison — a trivial most-frequent floor, the FRTMS threshold
     rules (onnesim.evaluate.threshold_baseline), logistic regression, random
     forest, gradient boosting. Reported with per-class precision/recall AND a
     full 6x6 confusion MATRIX (which faults get confused with which — the real
     story, per FINDINGS #3).
  3. REALISM STRESS TEST — how accuracy DEGRADES as we add sensor noise, shorten
     the observation window, and reduce fault severity. A benchmark that stays
     pinned at ~1.0 under stress is still too easy, and the harness quantifies it.

Reuse (nothing re-implemented, nothing edited):
  - features       : onnesim.ml_baseline.extract_features  (the existing approach)
  - detection score: onnesim.evaluate.score                (same dict shape as the agent)
  - rules baseline : onnesim.evaluate.threshold_baseline   (the "what exists today" bar)
  - fingerprint    : onnesim.virtual_clone.learn_fingerprint (real BlueFors noise)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import cryo_engine as CE
from . import evaluate as E
from . import ml_baseline as MB
from . import virtual_clone as VC

# Canonical label order (rows/cols of every confusion matrix are in THIS order).
FAULT_CLASSES: list[str] = list(CE.FAULT_CLASSES)
NORMAL = "normal"

# Physically-ambiguous cluster (docs/FINDINGS.md #3): a helium leak, wiring heat
# ingress, and a (currently underpowered) magnet quench all present as "cold /
# adjacent stages warm slowly, flow ~flat", so they are genuinely hard to tell
# apart from temperature alone. We MEASURE this confusion, and also report a
# "grouped" score that forgives within-cluster mistakes to quantify how much of
# the error is just this known ambiguity vs. real model failure.
AMBIGUOUS_GROUP: tuple[str, ...] = ("helium_leak", "wiring_heat_ingress", "magnet_quench")
DEFAULT_FINGERPRINT_DIR = "data/real/bluefors_cryometrics_sample"


# --------------------------------------------------------------------------- #
#  Scenario specs + rendering (the honest, seed-addressable dataset)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    """A fully seed-addressed scenario: everything about it derives from `seed`,
    so holding out by index == holding out by seed (disjoint random draws)."""
    seed: int
    fault_class: str
    severity: float
    onset_frac: float


def sample_specs(n: int, base_seed: int = 0) -> list[Spec]:
    """Sample n scenarios, each with its OWN seed. cryo_engine.sample_scenario is
    used unchanged; we just pin one rng per index so the whole spec (class,
    severity, onset) AND the later telemetry noise are tied to a unique seed."""
    specs = []
    for i in range(n):
        seed = base_seed + i
        sc = CE.sample_scenario(np.random.default_rng(seed))
        specs.append(Spec(seed, str(sc.fault_class), float(sc.severity), float(sc.onset_frac)))
    return specs


def render(spec: Spec, cfg: CE.EngineConfig, hours: float = 6.0, dt_min: float = 5.0,
           sev_scale: float = 1.0, extra_noise: float = 0.0, window_frac: float = 1.0) -> dict:
    """Render one spec to FRTMS telemetry via the UNCHANGED engine, with optional
    realism stressors applied to the TEST distribution only:
      sev_scale   < 1 : weaker fault (regenerated at scaled severity)
      window_frac < 1 : shorter observation window (truncate to a prefix)
      extra_noise > 0 : extra relative sensor noise on temps + flow
    Stressors are deterministic in `seed`, so a stressed run is reproducible."""
    scenario = CE.Scenario(spec.fault_class, spec.severity * sev_scale, spec.onset_frac)
    cols = CE.simulate(scenario, cfg, hours=hours, dt_min=dt_min, seed=spec.seed)

    if window_frac < 1.0:
        n = len(cols["t_s"])
        keep = max(2, int(round(n * window_frac)))
        cols = {k: np.asarray(v)[:keep] for k, v in cols.items()}

    if extra_noise > 0.0:
        rng = np.random.default_rng(spec.seed + 9_973)  # independent of the engine's own noise
        for ch in [f"temp{i}_T" for i in range(1, 8)] + ["flowmeter"]:
            y = np.asarray(cols[ch], dtype=float)
            cols[ch] = y * (1.0 + extra_noise * rng.standard_normal(y.shape))
    return cols


def build_X(cols_list: list[dict]) -> np.ndarray:
    """Feature matrix via the EXISTING ml_baseline approach (120 feats/window)."""
    return np.asarray([MB.extract_features(c)[0] for c in cols_list])


def load_fingerprint(path: str | None) -> dict | None:
    """Learn the real BlueFors noise fingerprint if available; else fall back to
    the engine's modest built-in noise (reported honestly either way)."""
    if not path:
        return None
    try:
        return VC.learn_fingerprint(path)
    except Exception as exc:  # noqa: BLE001 - fingerprint is optional, never fatal
        print(f"[benchmark] fingerprint unavailable ({exc}); using engine's built-in noise")
        return None


# --------------------------------------------------------------------------- #
#  Models (all reuse existing code; every model yields a full multiclass vector)
# --------------------------------------------------------------------------- #
def fit_ml_models(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0,
                  include_strong: bool = False) -> dict[str, object]:
    """Fit each ml_baseline estimator as a SINGLE multiclass classifier over all
    6 classes (clean confusion matrix). Factories come from ml_baseline.make_models.
    include_strong adds LightGBM + TabPFN when installed (stronger opponents)."""
    fitted = {}
    for name, make in MB.make_models(seed, include_strong=include_strong).items():
        est = make()
        est.fit(X_tr, y_tr)
        fitted[name] = est
    return fitted


def predict_threshold(cols_list: list[dict]) -> np.ndarray:
    """FRTMS threshold rules (onnesim.evaluate.threshold_baseline) as a model.
    Note: the rules can only emit a subset of classes — that limitation is real
    and is exactly what the ML/agent must beat."""
    out = []
    for cols in cols_list:
        v = E.threshold_baseline(cols)
        out.append(v["fault_class"] if v["fault_detected"] else NORMAL)
    return np.asarray(out, dtype=object)


def predict_engineered(cols_list: list[dict]) -> np.ndarray:
    """Engineered physics-rule baseline (onnesim.evaluate.engineered_baseline) as a
    model: rate-of-change + delta-pressure/flow rules on the discriminating channels.
    This is the "what a good cryo engineer builds first" bar — stronger than the
    fixed-cutoff FRTMS rules, and the honest reference the LLM agent must beat.
    Uses NO training labels (thresholds are the real sensor-noise floors)."""
    out = []
    for cols in cols_list:
        v = E.engineered_baseline(cols)
        out.append(v["fault_class"] if v["fault_detected"] else NORMAL)
    return np.asarray(out, dtype=object)


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def _macro_f1(y_true, y_pred, labels) -> float:
    present = [c for c in labels if c in set(map(str, y_true))]
    if not present:
        return 0.0
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="macro", zero_division=0)
    return float(f1)


def _grouped(y):
    """Collapse the ambiguous thermal cluster to one label (forgive its confusion)."""
    g = "thermal_ambiguous"
    return np.asarray([g if str(v) in AMBIGUOUS_GROUP else str(v) for v in y], dtype=object)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All the numbers for one model on one test set:
      - detection metrics via evaluate.score() (SAME shape as the agent eval)
      - per-class precision/recall/f1/support
      - macro-F1 (strict) and grouped macro-F1 (ambiguous cluster forgiven)
      - the full 6x6 confusion matrix (truth rows x pred cols, FAULT_CLASSES order)
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    # detection/classification, reusing evaluate.score() exactly
    rows = [{"truth_class": str(t), "pred_detected": bool(str(p) != NORMAL), "pred_class": str(p)}
            for t, p in zip(y_true, y_pred)]
    score = E.score(rows)

    # per-class P/R/F1/support
    labels_present = [c for c in FAULT_CLASSES if c in set(map(str, y_true))]
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, zero_division=0)
    per_class = {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                     "f1": round(float(f1[i]), 3), "support": int(sup[i])}
                 for i, c in enumerate(labels_present)}

    cm = confusion_matrix(list(map(str, y_true)), list(map(str, y_pred)), labels=FAULT_CLASSES)
    overall_acc = float(np.mean(np.asarray(list(map(str, y_true))) ==
                                np.asarray(list(map(str, y_pred)))))

    return {
        "score": score,                                   # evaluate.score() dict
        "overall_multiclass_accuracy": round(overall_acc, 3),
        "macro_f1": round(_macro_f1(y_true, y_pred, FAULT_CLASSES), 3),
        "grouped_macro_f1": round(_macro_f1(_grouped(y_true), _grouped(y_pred),
                                            [c for c in FAULT_CLASSES if c not in AMBIGUOUS_GROUP]
                                            + ["thermal_ambiguous"]), 3),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": list(FAULT_CLASSES),
    }


def top_confusions(cm: list[list[int]], labels: list[str], k: int = 5) -> list[dict]:
    """Largest off-diagonal (truth != pred) cells — the biggest confusions."""
    arr = np.asarray(cm)
    out = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and arr[i, j] > 0:
                out.append({"truth": labels[i], "pred": labels[j], "count": int(arr[i, j])})
    out.sort(key=lambda d: d["count"], reverse=True)
    return out[:k]


# --------------------------------------------------------------------------- #
#  Label-noise robustness (another "too easy" probe)
# --------------------------------------------------------------------------- #
def add_label_noise(y: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Flip a fraction of labels to a uniformly-random DIFFERENT class. If test
    accuracy barely moves, the task is so separable that even corrupted training
    labels don't matter — a strong 'too easy' signal."""
    if frac <= 0:
        return np.asarray(y, dtype=object)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=object).copy()
    idx = rng.random(len(y)) < frac
    for i in np.where(idx)[0]:
        alt = [c for c in FAULT_CLASSES if c != str(y[i])]
        y[i] = rng.choice(alt)
    return y


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def _jsonify(o):
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run(n: int = 360, hours: float = 6.0, dt_min: float = 5.0, test_frac: float = 0.3,
        base_seed: int = 0, fingerprint_dir: str | None = DEFAULT_FINGERPRINT_DIR,
        seed: int = 0) -> dict:
    """Full benchmark: honest split -> multi-model table -> stress test -> verdict."""
    cfg = CE.EngineConfig(fingerprint=load_fingerprint(fingerprint_dir))

    # ---- honest hold-out BY SEED --------------------------------------------
    specs = sample_specs(n, base_seed)
    n_test = max(1, int(round(n * test_frac)))
    train_specs, test_specs = specs[:-n_test], specs[-n_test:]
    train_seeds = {s.seed for s in train_specs}
    test_seeds = {s.seed for s in test_specs}
    assert train_seeds.isdisjoint(test_seeds), "seed leak between train and test"

    train_cols = [render(s, cfg, hours, dt_min) for s in train_specs]
    test_cols = [render(s, cfg, hours, dt_min) for s in test_specs]
    X_tr = build_X(train_cols)
    X_te = build_X(test_cols)
    y_tr = np.asarray([s.fault_class for s in train_specs], dtype=object)
    y_te = np.asarray([s.fault_class for s in test_specs], dtype=object)

    classes, counts = np.unique(np.asarray([s.fault_class for s in specs]), return_counts=True)

    # ---- main multi-model comparison ----------------------------------------
    fitted = fit_ml_models(X_tr, y_tr, seed)
    models: dict[str, dict] = {}
    for name, est in fitted.items():
        models[name] = evaluate_predictions(y_te, est.predict(X_te))
    models["threshold_rules"] = evaluate_predictions(y_te, predict_threshold(test_cols))
    models["engineered_rules"] = evaluate_predictions(y_te, predict_engineered(test_cols))

    # nicely ordered display: trivial -> rules -> learned
    order = ["most_frequent", "threshold_rules", "engineered_rules", "logreg",
             "random_forest", "hist_gbdt"]
    models = {k: models[k] for k in order if k in models}

    # pick the strongest LEARNED model by detection F1 (tie-break macro-F1)
    learned = [k for k in models if k not in ("most_frequent", "threshold_rules",
                                              "engineered_rules")]
    best = max(learned, key=lambda k: (models[k]["score"]["f1"], models[k]["macro_f1"]))

    # ---- realism stress test (train clean, test degraded) -------------------
    stress_models = {k: fitted[k] for k in ("random_forest", "hist_gbdt") if k in fitted}
    stress = {
        "noise": _stress_axis(stress_models, test_specs, cfg, hours, dt_min, y_te,
                              "extra_noise", [0.0, 0.02, 0.05, 0.10, 0.20]),
        "window_frac": _stress_axis(stress_models, test_specs, cfg, hours, dt_min, y_te,
                                    "window_frac", [1.0, 0.5, 0.25, 0.1]),
        "severity_scale": _stress_axis(stress_models, test_specs, cfg, hours, dt_min, y_te,
                                       "sev_scale", [1.0, 0.7, 0.5, 0.3, 0.15]),
    }

    # ---- confusion UNDER STRESS (the real story) ----------------------------
    # On clean default data the matrix is ~perfectly diagonal, so "which faults get
    # confused with which" is only visible once the task is made hard. Capture the
    # best model's confusion at a representative in-band noise level so the report
    # shows real confusion structure, not an empty off-diagonal.
    stress_noise_level = 0.10
    stressed_cols = [render(s, cfg, hours, dt_min, extra_noise=stress_noise_level)
                     for s in test_specs]
    stressed_pred = fitted[best].predict(build_X(stressed_cols))
    confusion_under_stress = evaluate_predictions(y_te, stressed_pred)
    confusion_under_stress["noise_level"] = stress_noise_level

    # ---- label-noise robustness sweep ---------------------------------------
    label_noise = {}
    for frac in (0.0, 0.1, 0.2):
        y_noisy = add_label_noise(y_tr, frac, seed=seed + 1)
        est = MB.make_models(seed)[best]()
        est.fit(X_tr, y_noisy)
        m = evaluate_predictions(y_te, est.predict(X_te))
        label_noise[f"{frac:.2f}"] = {"detection_f1": m["score"]["f1"], "macro_f1": m["macro_f1"]}

    verdict = _verdict(models[best], stress, best)

    return {
        "config": {"n": n, "hours": hours, "dt_min": dt_min, "test_frac": test_frac,
                   "n_train": len(train_specs), "n_test": len(test_specs),
                   "fingerprint": bool(cfg.fingerprint is not None),
                   "fingerprint_dir": fingerprint_dir if cfg.fingerprint is not None else None,
                   "holdout": "by-seed (train/test seed sets disjoint)"},
        "class_balance": {str(c): int(n_) for c, n_ in zip(classes, counts)},
        "models": models,
        "best_learned_model": best,
        "top_confusions": {k: top_confusions(v["confusion_matrix"], v["confusion_labels"])
                           for k, v in models.items()},
        "confusion_under_stress": confusion_under_stress,
        "top_confusions_under_stress": top_confusions(
            confusion_under_stress["confusion_matrix"],
            confusion_under_stress["confusion_labels"]),
        "stress_test": stress,
        "label_noise_robustness": label_noise,
        "verdict": verdict,
    }


def _stress_axis(stress_models: dict, test_specs: list[Spec], cfg, hours, dt_min,
                 y_te: np.ndarray, kwarg: str, levels: list[float]) -> list[dict]:
    """Evaluate fitted models on the test set re-rendered at each stress level."""
    rows = []
    for lvl in levels:
        cols = [render(s, cfg, hours, dt_min, **{kwarg: lvl}) for s in test_specs]
        X = build_X(cols)
        entry = {"level": lvl}
        for name, est in stress_models.items():
            m = evaluate_predictions(y_te, est.predict(X))
            entry[name] = {"detection_f1": m["score"]["f1"], "macro_f1": m["macro_f1"],
                           "overall_acc": m["overall_multiclass_accuracy"]}
        rows.append(entry)
    return rows


def _verdict(best_clean: dict, stress: dict, best_name: str) -> dict:
    """Automatic honesty check: is the benchmark actually hard yet?
    Target (docs/FINDINGS.md #1): strong models should land ~0.75-0.85, and
    accuracy should DEGRADE under realistic stress. If clean F1 >= 0.95 and it
    barely moves under heavy noise, the benchmark is still too easy — say so."""
    clean_f1 = best_clean["score"]["f1"]
    clean_macro = best_clean["macro_f1"]

    def worst(axis, key):
        vals = [lvl[best_name][key] for lvl in stress[axis] if best_name in lvl]
        return min(vals) if vals else clean_f1

    heavy_noise = stress["noise"][-1][best_name]["macro_f1"] if best_name in stress["noise"][-1] else clean_macro
    short_window = stress["window_frac"][-1][best_name]["macro_f1"] if best_name in stress["window_frac"][-1] else clean_macro
    low_sev = stress["severity_scale"][-1][best_name]["macro_f1"] if best_name in stress["severity_scale"][-1] else clean_macro
    worst_macro = min(worst("noise", "macro_f1"), worst("window_frac", "macro_f1"),
                      worst("severity_scale", "macro_f1"))

    too_easy_clean = clean_f1 >= 0.95
    robust_to_stress = worst_macro >= 0.90  # barely degrades -> still too easy
    degrades_into_band = worst_macro <= 0.88  # stress pushes it into/below target

    # Two honest facts, stated separately: (a) the DEFAULT operating point, and
    # (b) behavior under realistic stress. Don't let one hide the other.
    if too_easy_clean and robust_to_stress:
        label = ("TOO EASY — perfect/near-perfect at default settings AND stays >=0.90 "
                 "under heavy stress. Benchmark is not hard enough yet.")
    elif too_easy_clean and degrades_into_band:
        label = ("DEFAULT IS TOO EASY (clean F1 ~1.0) BUT degrades into/below the 0.75-0.85 "
                 "target band under realistic stress. Ship the STRESSED configs as the real "
                 "benchmark; the clean default is not a meaningful test.")
    elif 0.70 <= clean_macro <= 0.88:
        label = "IN TARGET BAND at default settings — good."
    else:
        label = "MIXED — see numbers"

    return {
        "best_model": best_name,
        "clean_detection_f1": clean_f1,
        "clean_macro_f1": clean_macro,
        "default_setting_is_trivial": bool(too_easy_clean),
        "macro_f1_heavy_noise_0.20": round(heavy_noise, 3),
        "macro_f1_short_window_0.10": round(short_window, 3),
        "macro_f1_low_severity_0.15": round(low_sev, 3),
        "worst_case_macro_f1_under_stress": round(worst_macro, 3),
        "target_band": "0.75-0.85 (docs/FINDINGS.md #1)",
        "label": label,
    }
