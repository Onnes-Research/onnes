"""Tests for the staged cooldown produced by onnesim.model.simulate()."""
from __future__ import annotations

import numpy as np
import pytest

from onnesim import FridgeParams, simulate
from onnesim import constants as C

from .conftest import run_sim, DURATION_S, DT_S


# Warm -> cold stage order that a real dry dilution fridge must exhibit.
EXPECTED_ORDER = ["50K", "4K", "still", "coldplate", "mxc"]


def test_simulate_returns_expected_shape_and_keys(normal_sim):
    sim = normal_sim
    assert set(sim.keys()) == {"t_s", "T", "stages"}
    n_steps = int(DURATION_S / DT_S) + 1
    assert sim["T"].shape == (n_steps, C.N_STAGES)
    assert sim["t_s"].shape == (n_steps,)
    assert list(sim["stages"]) == EXPECTED_ORDER


def test_time_axis_is_monotonic_and_spaced(normal_sim):
    t = normal_sim["t_s"]
    assert t[0] == 0.0
    dt = np.diff(t)
    assert np.allclose(dt, DT_S)
    assert t[-1] == pytest.approx(DURATION_S)


def test_no_nan_or_inf_anywhere(normal_sim):
    assert np.all(np.isfinite(normal_sim["T"])), "cooldown history contains NaN/inf"
    assert np.all(np.isfinite(normal_sim["t_s"]))


def test_stages_end_separated_and_ordered(normal_end):
    """Stages must end STRICTLY ordered 50K > 4K > still > coldplate > mxc."""
    end = normal_end
    for i in range(C.N_STAGES - 1):
        assert end[i] > end[i + 1], (
            f"stage {EXPECTED_ORDER[i]}={end[i]:.4g} K is not warmer than "
            f"{EXPECTED_ORDER[i + 1]}={end[i + 1]:.4g} K — stages not separated/ordered"
        )


def test_stages_end_well_separated(normal_end):
    """Adjacent stages should be clearly separated, not merely a hair apart.

    Each stage should end at least ~20% colder than the stage above it. This is a
    relational check (no PLACEHOLDER magnitude baked in) that guards against the
    stages collapsing toward a common temperature.
    """
    end = normal_end
    for i in range(C.N_STAGES - 1):
        assert end[i + 1] < 0.8 * end[i], (
            f"{EXPECTED_ORDER[i + 1]} ({end[i + 1]:.4g} K) not clearly below "
            f"{EXPECTED_ORDER[i]} ({end[i]:.4g} K)"
        )


def test_temps_within_physical_sanity_bounds(normal_sim, normal_end):
    """Every temperature stays within loose, model-agnostic physical bounds.

    Lower bound: temperatures are absolute, so must stay > 0 K. Upper bound:
    nothing can exceed the room-temperature start (plus a small numerical margin
    for the noiseless ODE overshoot near t=0).
    """
    hist = normal_sim["T"]
    assert hist.min() > 0.0, "temperature went to/below absolute zero"
    assert hist.max() <= C.T_ROOM * 1.05, "temperature exceeded room-temp start"

    # End state: coldest stage (mxc) must reach the sub-Kelvin regime a real
    # dilution fridge operates in; warmest stage stays well below room temp.
    end = normal_end
    assert end[-1] < 1.0, f"mxc did not reach sub-Kelvin (ended {end[-1]:.4g} K)"
    assert end[0] < C.T_ROOM, "50K plate never cooled below room temperature"


def test_stages_cool_from_room_temperature(normal_sim):
    """Every stage must start at the room-temperature initial condition and end
    below it — i.e. a real cooldown happened, not a flat line."""
    hist = normal_sim["T"]
    assert np.allclose(hist[0], C.T_ROOM)
    assert np.all(hist[-1] < hist[0]), "a stage failed to cool below its start"


def test_simulation_is_deterministic():
    """The ODE core carries no RNG; identical inputs must give identical output."""
    a = run_sim(None)["T"]
    b = run_sim(None)["T"]
    assert np.array_equal(a, b), "simulate() is not deterministic for fixed inputs"


def test_custom_short_duration_scales_step_count():
    """duration_s / dt_s controls the number of integration steps."""
    sim = simulate(FridgeParams(), duration_s=3600.0, dt_s=60.0)
    assert sim["T"].shape[0] == int(3600.0 / 60.0) + 1
