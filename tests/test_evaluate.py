"""Tests for onnesim.evaluate.score() — the CryoOpsBench metric math.

These use hand-built (truth, pred) rows with KNOWN confusion-matrix counts, so
the precision/recall/f1/accuracy arithmetic is checked against values computed by
hand (and cross-checked below against scikit-learn where applicable).
"""
from __future__ import annotations

import math

import pytest

from onnesim import evaluate as E


def _row(truth, detected, pred_class):
    return {"truth_class": truth, "pred_detected": detected, "pred_class": pred_class}


def test_score_counts_and_metrics_on_known_matrix():
    """Hand-built rows: tp=2, fp=1, tn=2, fn=1 (n=6).

    precision = 2/3, recall = 2/3, f1 = 2/3, detect_accuracy = 4/6 = 2/3,
    class_accuracy_on_detected = 1/2 (one of the two detected faults mislabeled).
    """
    rows = [
        _row("helium_leak", True, "helium_leak"),      # tp, class correct
        _row("magnet_quench", True, "helium_leak"),    # tp, class WRONG
        _row("heat_load_spike", False, "normal"),      # fn (missed fault)
        _row("normal", True, "helium_leak"),           # fp (false alarm)
        _row("normal", False, "normal"),               # tn
        _row("normal", False, "normal"),               # tn
    ]
    s = E.score(rows)

    assert s["counts"] == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    assert s["n"] == 6
    assert s["precision"] == round(2 / 3, 3) == 0.667
    assert s["recall"] == round(2 / 3, 3) == 0.667
    assert s["f1"] == round(2 / 3, 3) == 0.667
    assert s["detect_accuracy"] == round(4 / 6, 3) == 0.667
    assert s["class_accuracy_on_detected"] == 0.5


def test_score_perfect_detection():
    """All faults detected, no false alarms -> precision=recall=f1=1."""
    rows = [
        _row("helium_leak", True, "helium_leak"),
        _row("heat_load_spike", True, "heat_load_spike"),
        _row("normal", False, "normal"),
        _row("normal", False, "normal"),
    ]
    s = E.score(rows)
    assert s["counts"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
    assert s["precision"] == 1.0
    assert s["recall"] == 1.0
    assert s["f1"] == 1.0
    assert s["detect_accuracy"] == 1.0
    assert s["class_accuracy_on_detected"] == 1.0


def test_score_all_missed_faults():
    """Every fault missed -> recall 0, and (per code) precision 0 via max()."""
    rows = [
        _row("helium_leak", False, "normal"),
        _row("magnet_quench", False, "normal"),
    ]
    s = E.score(rows)
    assert s["counts"] == {"tp": 0, "fp": 0, "tn": 0, "fn": 2}
    assert s["recall"] == 0.0
    assert s["precision"] == 0.0
    assert s["f1"] == 0.0
    assert s["detect_accuracy"] == 0.0
    # No detections -> class accuracy denominator guarded to 1 -> 0/1 = 0.
    assert s["class_accuracy_on_detected"] == 0.0


def test_score_precision_recall_differ():
    """Asymmetric matrix so precision != recall, checked by hand.

    tp=1, fp=2, fn=0, tn=1 -> precision=1/3=0.333, recall=1/1=1.0,
    f1 = 2*(1/3*1)/(1/3+1) = (2/3)/(4/3) = 0.5.
    """
    rows = [
        _row("helium_leak", True, "helium_leak"),  # tp
        _row("normal", True, "helium_leak"),        # fp
        _row("normal", True, "heat_load_spike"),    # fp
        _row("normal", False, "normal"),            # tn
    ]
    s = E.score(rows)
    assert s["counts"] == {"tp": 1, "fp": 2, "tn": 1, "fn": 0}
    assert s["precision"] == round(1 / 3, 3) == 0.333
    assert s["recall"] == 1.0
    assert s["f1"] == 0.5


def test_score_f1_is_harmonic_mean():
    """Independently recompute f1 from the reported precision/recall."""
    rows = [
        _row("helium_leak", True, "helium_leak"),
        _row("magnet_quench", True, "magnet_quench"),
        _row("heat_load_spike", False, "normal"),
        _row("normal", True, "helium_leak"),
        _row("normal", False, "normal"),
    ]
    s = E.score(rows)
    p, r = s["precision"], s["recall"]
    expected_f1 = round(2 * p * r / (p + r), 3)
    assert s["f1"] == pytest.approx(expected_f1, abs=1e-3)


def test_score_matches_sklearn_on_detection():
    """Cross-check detect precision/recall/f1 against scikit-learn.

    Skips cleanly if scikit-learn (the [ml] extra) is not installed.
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rows = [
        _row("helium_leak", True, "helium_leak"),      # tp
        _row("magnet_quench", True, "magnet_quench"),  # tp
        _row("heat_load_spike", False, "normal"),      # fn
        _row("normal", True, "helium_leak"),           # fp
        _row("normal", False, "normal"),               # tn
        _row("normal", False, "normal"),               # tn
    ]
    y_true = [0 if r["truth_class"] == "normal" else 1 for r in rows]
    y_pred = [1 if r["pred_detected"] else 0 for r in rows]

    s = E.score(rows)
    assert s["precision"] == pytest.approx(
        sklearn_metrics.precision_score(y_true, y_pred), abs=1e-3)
    assert s["recall"] == pytest.approx(
        sklearn_metrics.recall_score(y_true, y_pred), abs=1e-3)
    assert s["f1"] == pytest.approx(
        sklearn_metrics.f1_score(y_true, y_pred), abs=1e-3)
    assert s["detect_accuracy"] == pytest.approx(
        sklearn_metrics.accuracy_score(y_true, y_pred), abs=1e-3)


def test_score_empty_rows_does_not_crash():
    """Guarded denominators (max(...,1)) must keep an empty dataset from dividing by zero."""
    s = E.score([])
    assert s["n"] == 1  # n is guarded to >=1 by the implementation
    assert s["precision"] == 0.0
    assert s["recall"] == 0.0
    assert s["f1"] == 0.0
    assert s["counts"] == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
