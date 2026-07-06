"""Shared pytest fixtures for the OnnesSim suite.

Design choices:
- A 24 h / 60 s cooldown is the canonical grid: empirically it is long enough
  that all five stages settle into the correct top-down ordering while running
  in well under a second (kept short so the suite is CI-cheap). A 6 h run is
  NOT settled (cold stages have not engaged), so we avoid it.
- Faults are injected with a matched (same params, grid, no-jitter) NORMAL
  baseline so each fault test is a controlled A/B: everything identical except
  the fault hook. Simulation is deterministic (RK4, no RNG in the ODE), so
  these baselines are exactly reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from onnesim import FridgeParams, simulate, FaultSpec, make_hook, emit
from onnesim import faults as F
from onnesim import constants as C


# --- canonical simulation grid -------------------------------------------- #
DURATION_S = 24.0 * 3600.0
DT_S = 60.0
ONSET_S = 0.45 * DURATION_S
SEVERITY = 0.8


def run_sim(spec: FaultSpec | None = None):
    """Run one deterministic cooldown on the canonical grid.

    spec=None -> a fault-free (normal) run. Otherwise the fault hook is wired in.
    """
    hook = make_hook(spec) if spec is not None else None
    return simulate(
        FridgeParams(),
        duration_s=DURATION_S,
        dt_s=DT_S,
        fault_hook=hook,
    )


def make_spec(fault_class: str, severity: float = SEVERITY, target_stage: int = -1) -> FaultSpec:
    """Build a FaultSpec on the canonical grid with a mid-run onset."""
    return FaultSpec(
        fault_class=fault_class,
        onset_s=ONSET_S,
        severity=severity,
        target_stage=target_stage,
    )


@pytest.fixture(scope="session")
def stage_index():
    """Map stage name -> index, straight from constants (single source of truth)."""
    return {name: i for i, name in enumerate(C.STAGE_NAMES)}


@pytest.fixture(scope="session")
def normal_sim():
    """A cached fault-free cooldown, reused as the baseline across tests."""
    return run_sim(None)


@pytest.fixture(scope="session")
def normal_end(normal_sim):
    """Final per-stage temperatures [K] of the normal cooldown."""
    return normal_sim["T"][-1]
