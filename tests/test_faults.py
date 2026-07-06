"""Tests for fault injection: each fault must raise the physically-correct stage
end-temperature relative to a matched fault-free baseline.

All comparisons are RELATIVE (fault vs normal on the same grid/params), so they
do not depend on the PLACEHOLDER magnitudes in constants.py — only on the sign
of the effect, which is the invariant a benchmark label must uphold.
"""
from __future__ import annotations

import numpy as np
import pytest

from onnesim import FaultSpec, make_hook, simulate, FridgeParams
from onnesim import faults as F
from onnesim import constants as C

from .conftest import run_sim, make_spec, DURATION_S, DT_S, ONSET_S


def _end_temps(spec: FaultSpec) -> np.ndarray:
    return run_sim(spec)["T"][-1]


def test_helium_leak_raises_cold_stages(normal_end):
    """A helium leak fades dilution cooling -> still/coldplate/mxc end warmer."""
    leak_end = _end_temps(make_spec("helium_leak"))
    for stage in (F.STILL, F.COLDPLATE, F.MXC):
        assert leak_end[stage] > normal_end[stage], (
            f"helium_leak did not warm {C.STAGE_NAMES[stage]}: "
            f"normal={normal_end[stage]:.5g} leak={leak_end[stage]:.5g}"
        )


def test_heat_load_spike_raises_targeted_cold_stage(normal_end):
    """A heat-load spike on the mixing chamber raises the mxc end-temperature."""
    spike_end = _end_temps(make_spec("heat_load_spike", target_stage=F.MXC))
    assert spike_end[F.MXC] > normal_end[F.MXC], (
        f"heat_load_spike(MXC) did not warm mxc: "
        f"normal={normal_end[F.MXC]:.5g} spike={spike_end[F.MXC]:.5g}"
    )


def test_heat_load_spike_default_targets_coldplate(normal_end):
    """With target_stage=-1 the spike defaults onto the cold plate (per faults.py)."""
    spec = FaultSpec("heat_load_spike", onset_s=ONSET_S, severity=0.8, target_stage=-1)
    spike_end = _end_temps(spec)
    assert spike_end[F.COLDPLATE] > normal_end[F.COLDPLATE], (
        f"heat_load_spike(default) did not warm coldplate: "
        f"normal={normal_end[F.COLDPLATE]:.5g} spike={spike_end[F.COLDPLATE]:.5g}"
    )


def test_magnet_quench_raises_4k_stage(normal_end):
    """A magnet quench dumps load onto the 4K flange (temp2 / index FOURK)."""
    quench_end = _end_temps(make_spec("magnet_quench", target_stage=F.FOURK))
    assert quench_end[F.FOURK] > normal_end[F.FOURK], (
        f"magnet_quench did not warm the 4K stage: "
        f"normal={normal_end[F.FOURK]:.5g} quench={quench_end[F.FOURK]:.5g}"
    )


def test_normal_fault_is_a_noop(normal_end):
    """fault_class='normal' must not perturb the cooldown at all."""
    spec = FaultSpec("normal", onset_s=DURATION_S * 2, severity=0.0, target_stage=F.MXC)
    end = _end_temps(spec)
    assert np.allclose(end, normal_end), "normal fault changed the trajectory"


def test_higher_severity_gives_larger_excursion(normal_end):
    """Monotonicity: a more severe heat-load spike warms the stage more."""
    mild = _end_temps(make_spec("heat_load_spike", severity=0.4, target_stage=F.MXC))
    severe = _end_temps(make_spec("heat_load_spike", severity=1.0, target_stage=F.MXC))
    assert severe[F.MXC] > mild[F.MXC] > normal_end[F.MXC]


def test_faulted_cooldown_has_no_nan_or_inf():
    """Fault hooks must not destabilize the integrator."""
    for fc in ("helium_leak", "heat_load_spike", "magnet_quench"):
        stage = F.FOURK if fc == "magnet_quench" else F.MXC
        sim = run_sim(make_spec(fc, target_stage=stage))
        assert np.all(np.isfinite(sim["T"])), f"{fc} produced NaN/inf"


def test_make_hook_returns_per_stage_arrays():
    """A hook returns (extra_load, cooling_scale), each length N_STAGES."""
    hook = make_hook(make_spec("heat_load_spike", target_stage=F.MXC))
    extra, scale = hook(ONSET_S + 3600.0, np.full(C.N_STAGES, 1.0))
    assert extra.shape == (C.N_STAGES,)
    assert scale.shape == (C.N_STAGES,)


def test_hook_inactive_before_onset():
    """Before onset there is no extra load and cooling is unscaled."""
    hook = make_hook(make_spec("heat_load_spike", target_stage=F.MXC))
    extra, scale = hook(0.0, np.full(C.N_STAGES, 1.0))
    assert np.allclose(extra, 0.0)
    assert np.allclose(scale, 1.0)
