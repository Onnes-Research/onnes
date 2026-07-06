"""Tests for the real Cryowala physics port + ML surrogate."""
import numpy as np
import pytest

from onnesim import cryowala_physics as CP


def test_reproduces_calibrated_base_temps():
    """At the fit set-point, T_funcs must reproduce the real fridge base temps (t0)."""
    temps = CP.all_stage_temps(CP.NOMINAL_LOADS_W)
    # calibrated t0 from CryowalaCore: [42.0, 3.41, 1.205, 0.163, 0.0165] K
    expected = np.array([42.0111, 3.4103, 1.2052, 0.16336, 0.016477])
    assert np.allclose(temps, expected, rtol=0.02), f"got {temps}"


def test_stage_ordering_at_nominal():
    """Nominal fridge is warm->cold ordered."""
    t = CP.all_stage_temps(CP.NOMINAL_LOADS_W)
    assert np.all(np.diff(t) < 0), f"stages not ordered: {t}"


def test_mxc_heats_with_added_load():
    """Adding heat to the MXC must warm it (monotone, physical response)."""
    base = CP.NOMINAL_LOADS_W.copy()
    t0 = CP.all_stage_temps(base)[4]
    hot = base.copy(); hot[4] += 20e-6  # +20 uW
    t1 = CP.all_stage_temps(hot)[4]
    assert t1 > t0, f"MXC did not warm: {t0} -> {t1}"


def test_is_physical_filter():
    """is_physical rejects negative / unordered temperature vectors."""
    assert CP.is_physical(np.array([46, 4, 1, 0.1, 0.01]))
    assert not CP.is_physical(np.array([46, 4, 1, 0.1, -0.004]))  # negative
    assert not CP.is_physical(np.array([46, 4, 1, 0.01, 0.1]))    # unordered


def test_dataset_is_physically_valid():
    """make_dataset must return only physical samples."""
    S = pytest.importorskip("onnesim.ml_surrogate")
    X, Y = S.make_dataset(500, seed=0)
    assert len(X) > 0
    assert np.all(Y > 0), "unphysical negative temps leaked into dataset"
    assert np.all(np.diff(Y, axis=1) < 0), "unordered stages leaked into dataset"


def test_surrogate_learns_map_quickly():
    """A short training run must already drive error well below the raw-fit failure.
    (Full accuracy ~0.01% needs the full 2000-epoch run; here we just prove it learns.)"""
    pytest.importorskip("torch")
    S = pytest.importorskip("onnesim.ml_surrogate")
    r = S.train_surrogate(n_train=4000, n_val=1000, epochs=200, batch=1024,
                          seed=0, verbose=False)
    # after 200 epochs on valid data, mean error should be < 5% (full run reaches ~0.01%)
    assert r["mean_rel_err"] < 0.05, f"surrogate not learning: {r['mean_rel_err']}"
