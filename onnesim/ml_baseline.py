"""
OnnesSim classical ML baselines for CryoOpsBench (the "trained ML" table column).

⚠️ These models are TRAINED ON PLACEHOLDER-PHYSICS SYNTHETIC DATA — NOT validated
against real hardware. Every physical constant in constants.py is a *plausible*
placeholder (see README "Honest status" + PHYSICS_NOTES.md). So the metrics below
measure how separable the *simulator's* injected fault signatures are — not how a
classifier would do on a real Bluefors/LD400 fridge. Do not cite as hardware results.

What this provides (the honest middle bar between the threshold baseline and the LLM agent):
  - feature extraction per telemetry window over the VERIFIED FRTMS schema channels
    (temp1_T..temp8_T, flowmeter, p1..p6): start, end, min, max, mean, slope,
    pct_change, std  -> 15 channels x 8 stats = 120 features per scenario.
  - a two-stage classical model:
        detector   = binary  {normal vs fault}
        classifier = multiclass over the 7 fault classes (trained on faults only)
  - evaluation that REUSES onnesim.evaluate.score(), so the reported metrics are the
    exact same dict shape as the agent and threshold baseline -> directly comparable.

Split is BY SCENARIO: one feature vector == one scenario, so an ordinary row-wise
train/test split (or k-fold) can never leak windows across the boundary.
"""
from __future__ import annotations
import csv
import os

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate as E

# Verified FRTMS schema channels used as features. Status bits (cpa_status,
# turbo_status) are constant step signals and are intentionally excluded.
FEATURE_CHANNELS = (
    [f"temp{i}_T" for i in range(1, 9)] + ["flowmeter"] + [f"p{i}" for i in range(1, 7)]
)
# Per-channel summary statistics over the window.
STAT_NAMES = ["start", "end", "min", "max", "mean", "slope", "pct_change", "std"]

NORMAL = "normal"


# --- feature engineering --------------------------------------------------- #
def extract_features(cols: dict) -> tuple[np.ndarray, list[str]]:
    """Summarize one telemetry window (channel -> array) into a flat feature vector.

    Features per channel: start, end, min, max, mean, slope (least-squares over
    time normalized to [0,1]), pct_change (end vs start), std.
    """
    t = np.asarray(cols["t_s"], dtype=float)
    span = float(t[-1] - t[0]) or 1.0
    tnorm = (t - t[0]) / span  # 0..1 so slope has consistent units across runs

    feats: list[float] = []
    names: list[str] = []
    for ch in FEATURE_CHANNELS:
        y = np.asarray(cols[ch], dtype=float)
        start, end = float(y[0]), float(y[-1])
        slope = float(np.polyfit(tnorm, y, 1)[0]) if np.ptp(tnorm) > 0 else 0.0
        pct_change = (end - start) / (abs(start) + 1e-9)
        feats.extend([start, end, float(np.min(y)), float(np.max(y)),
                      float(np.mean(y)), slope, pct_change, float(np.std(y))])
        names.extend(f"{ch}_{s}" for s in STAT_NAMES)
    return np.nan_to_num(np.asarray(feats, dtype=float)), names


def load_cols(path: str) -> dict:
    """Load one scenario CSV into {channel_name: np.ndarray} (matches scripts/evaluate.py)."""
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {name: data[:, i] for i, name in enumerate(header)}


def build_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Read labels.csv + every scenario CSV -> (X, y_class, scenario_ids, feature_names).

    X has one row per scenario, so callers split BY SCENARIO for free.
    """
    labels_path = os.path.join(dataset_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"no labels.csv in {dataset_dir}. Run scripts/generate_dataset.py --out {dataset_dir} first."
        )
    with open(labels_path) as fh:
        manifest = list(csv.DictReader(fh))

    X, y, ids, feat_names = [], [], [], None
    for row in manifest:
        cols = load_cols(os.path.join(dataset_dir, row["csv"]))
        vec, names = extract_features(cols)
        feat_names = feat_names or names
        X.append(vec)
        y.append(row["fault_class"])
        ids.append(row["csv"])
    return np.asarray(X), np.asarray(y, dtype=object), ids, feat_names


# --- optional stronger opponents ------------------------------------------- #
# Reviewers (all four) asked for baselines beyond a random forest: gradient
# boosting (LightGBM) and a tabular foundation model (TabPFN). Both are optional
# imports so the suite still runs if they are absent; availability is reported.
def _lightgbm_factory(seed: int):
    try:
        from lightgbm import LGBMClassifier
    except Exception:  # noqa: BLE001
        return None
    return lambda: LGBMClassifier(n_estimators=400, random_state=seed, n_jobs=-1,
                                  verbose=-1)


def _tabpfn_factory(seed: int):
    """TabPFN-3 tabular foundation model (pip 'tabpfn' v8.x == TabPFN-3, arXiv:2605.13986,
    Grinsztajn et al. 2026). In-context (no gradient training); .fit(X,y) stores the
    context. Local inference needs a one-time Prior Labs license token in TABPFN_TOKEN
    (the license permits research/internal evaluation, which a paper baseline is). CPU is
    fine at our scale. Returns None if the package is absent."""
    try:
        from tabpfn import TabPFNClassifier
    except Exception:  # noqa: BLE001
        return None
    def mk():
        try:
            return TabPFNClassifier(random_state=seed, ignore_pretraining_limits=True)
        except TypeError:
            return TabPFNClassifier()  # older signature
    return mk


def available_extra_models() -> dict:
    """Which optional strong baselines are importable in this environment."""
    return {"lightgbm": _lightgbm_factory(0) is not None,
            "tabpfn": _tabpfn_factory(0) is not None}


# --- models ---------------------------------------------------------------- #
def make_models(seed: int = 0, include_strong: bool = False) -> dict:
    """Fresh estimator factories to try. Each is used for BOTH stages (detect + classify).

    With include_strong=True, append the optional gradient-boosting and tabular
    foundation-model opponents when their libraries are installed."""
    models = {
        "most_frequent": lambda: DummyClassifier(strategy="most_frequent"),
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1),
        "hist_gbdt": lambda: HistGradientBoostingClassifier(random_state=seed),
    }
    if include_strong:
        lgbm = _lightgbm_factory(seed)
        tabpfn = _tabpfn_factory(seed)
        if lgbm is not None:
            models["lightgbm"] = lgbm
        if tabpfn is not None:
            models["tabpfn"] = tabpfn
    return models


# --- two-stage train / predict -------------------------------------------- #
def _fit_two_stage(make_est, X_tr: np.ndarray, y_tr: np.ndarray):
    """detector: normal-vs-fault on all rows; classifier: fault-type on fault rows only."""
    detector = make_est()
    detector.fit(X_tr, (y_tr != NORMAL).astype(int))

    fault_mask = y_tr != NORMAL
    classifier = make_est()
    classifier.fit(X_tr[fault_mask], y_tr[fault_mask])
    return detector, classifier


def _rows(detector, classifier, X: np.ndarray, y_true: np.ndarray) -> list[dict]:
    """Build score() rows. pred_class is only consulted by score() for detected faults."""
    det = detector.predict(X).astype(bool)
    cls = classifier.predict(X)
    return [{"truth_class": str(tc), "pred_detected": bool(d), "pred_class": str(pc)}
            for tc, d, pc in zip(y_true, det, cls)]


def evaluate_holdout(make_est, X, y, seed: int = 0, test_size: float = 0.25) -> dict:
    """Single stratified BY-SCENARIO holdout. Returns E.score() on the test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    detector, classifier = _fit_two_stage(make_est, X_tr, y_tr)
    return E.score(_rows(detector, classifier, X_te, y_te))


def evaluate_cv(make_est, X, y, folds: int = 5, seed: int = 0) -> dict:
    """Out-of-fold predictions across ALL scenarios (each scored by a model that never
    saw it), then a single E.score() over the full set — same coverage/n as the agent eval."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict] = []
    for tr_idx, te_idx in skf.split(X, y):
        detector, classifier = _fit_two_stage(make_est, X[tr_idx], y[tr_idx])
        rows.extend(_rows(detector, classifier, X[te_idx], y[te_idx]))
    return E.score(rows)


def run_all(X, y, seed: int = 0, test_size: float = 0.25, folds: int = 5) -> dict:
    """Train+evaluate every model with both a holdout and cross-validation.

    Returns a results dict; best model is picked by cross-validated detection F1
    (tie-broken by class_accuracy_on_detected), which is the most robust and is
    evaluated over all n scenarios for apples-to-apples comparison with the agent.
    """
    models = make_models(seed)
    holdout = {name: evaluate_holdout(mk, X, y, seed, test_size) for name, mk in models.items()}
    cross_val = {name: evaluate_cv(mk, X, y, folds, seed) for name, mk in models.items()}

    ranked = sorted(
        (n for n in models if n != "most_frequent"),
        key=lambda n: (cross_val[n]["f1"], cross_val[n]["class_accuracy_on_detected"]),
        reverse=True,
    )
    best = ranked[0]
    return {
        "holdout": holdout,
        "cross_val": cross_val,
        "best_model": best,
        "best_selection": "cross-validated detection f1 (tie-break class_accuracy_on_detected)",
    }
