"""Tests for the virtual clone (data-driven statistical twin of a real fridge)."""
import os
import numpy as np
import pytest

CLONE_DATA = "data/real/bluefors_cryometrics_sample"
have_data = os.path.isdir(CLONE_DATA)
pytestmark = pytest.mark.skipif(not have_data, reason="real BlueFors sample not present")


def test_clone_matches_real_std():
    """The clone's per-stage noise magnitude must match the real fridge's."""
    from onnesim import virtual_clone as VC
    fp = VC.learn_fingerprint(CLONE_DATA)
    v = VC.validate(fp, seed=0)
    for i in range(len(v["stages"])):
        rel = abs(v["clone_std"][i] - v["real_std"][i]) / (v["real_std"][i] + 1e-18)
        assert rel < 0.05, f"stage {v['stages'][i]} std off by {rel:.1%}"


def test_clone_matches_real_correlations():
    """The clone must reproduce the real cross-stage correlation structure
    (naive independent-Gaussian noise CANNOT do this)."""
    from onnesim import virtual_clone as VC
    fp = VC.learn_fingerprint(CLONE_DATA)
    v = VC.validate(fp, seed=0)
    err = np.abs(v["real_corr"] - v["clone_corr"])
    assert err.max() < 0.15, f"correlation structure not matched (max err {err.max():.2f})"


def test_clone_beats_independent_noise_on_correlation():
    """Prove the clone is BETTER than naive independent noise at matching the real
    fridge: independent noise has ~0 cross-correlation, real fridge has strong ones."""
    from onnesim import virtual_clone as VC
    fp = VC.learn_fingerprint(CLONE_DATA)
    # real off-diagonal correlations include strong ones (50K-MXC ~0.78)
    real_off = fp["corr"][np.triu_indices(len(fp["stages"]), 1)]
    assert np.abs(real_off).max() > 0.5, "expected strong real cross-correlations"
    # independent Gaussian would give ~0 for all; the clone reproduces the strong one
    v = VC.validate(fp, seed=1)
    clone_off = v["clone_corr"][np.triu_indices(len(fp["stages"]), 1)]
    # the clone's max correlation should be strong too (independent noise's ~0 fails)
    assert np.abs(clone_off).max() > 0.5
