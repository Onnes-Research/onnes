"""Tests for the FRTMS-schema telemetry emitter (onnesim.telemetry.emit).

The headline assertion is that emit() produces EXACTLY the verified FRTMS
columns and nothing else: temp1_T..temp8_T, p1..p6, flowmeter, cpa_status,
turbo_status, plus the t_s time axis.
"""
from __future__ import annotations

import numpy as np
import pytest

from onnesim import FaultSpec, emit
from onnesim import telemetry as TM
from onnesim import constants as C

from .conftest import run_sim, make_spec, DURATION_S


# The exact, ordered FRTMS column set the emitter must produce (besides t_s).
EXPECTED_CHANNELS = (
    [f"temp{i}_T" for i in range(1, 9)]   # temp1_T .. temp8_T
    + [f"p{i}" for i in range(1, 7)]      # p1 .. p6
    + ["flowmeter", "cpa_status", "turbo_status"]
)


@pytest.fixture(scope="module")
def normal_cols(normal_sim_module):
    spec = FaultSpec("normal", onset_s=DURATION_S * 2, severity=0.0, target_stage=C.N_STAGES - 1)
    return emit(normal_sim_module, spec, seed=1)


@pytest.fixture(scope="module")
def normal_sim_module():
    return run_sim(None)


def test_emit_produces_exactly_the_frtms_columns(normal_cols):
    """emit() must produce EXACTLY the verified FRTMS column set — nothing
    missing, nothing extra — plus the t_s time axis.

    We assert on the SET (not positional order): the FRTMS schema is a set of
    named DB columns, and the emitter is free to order them. This still fails
    crisply if any column is dropped, renamed, or added.
    """
    keys = list(normal_cols.keys())
    assert keys[0] == "t_s", "time axis t_s must be the first channel"

    got = set(keys)
    expected = set(["t_s"] + EXPECTED_CHANNELS)
    assert got == expected, (
        f"schema mismatch: missing={expected - got} extra={got - expected}"
    )
    # Exactly the right number of columns (guards against duplicate keys collapsing).
    assert len(keys) == len(expected) == 18  # t_s + 8 temp + 6 p + 3 status


def test_temperature_block_is_contiguous_after_time(normal_cols):
    """temp1_T..temp8_T appear as one ordered, contiguous block right after t_s."""
    keys = list(normal_cols.keys())
    assert keys[1:9] == [f"temp{i}_T" for i in range(1, 9)]


def test_emit_column_group_constants_match_schema():
    """The module's declared column groups equal the verified FRTMS names."""
    assert TM.TEMP_COLS == [f"temp{i}_T" for i in range(1, 9)]
    assert TM.PRESSURE_COLS == [f"p{i}" for i in range(1, 7)]
    assert TM.STATUS_COLS == ["flowmeter", "cpa_status", "turbo_status"]


def test_all_channels_same_length_as_time(normal_cols):
    n = normal_cols["t_s"].shape[0]
    for name, arr in normal_cols.items():
        assert np.asarray(arr).shape == (n,), f"channel {name} has wrong length"


def test_temperature_channels_are_finite_and_positive(normal_cols):
    for i in range(1, 9):
        ch = normal_cols[f"temp{i}_T"]
        assert np.all(np.isfinite(ch)), f"temp{i}_T not finite"
        assert np.all(ch > 0.0), f"temp{i}_T went non-positive"


def test_pressure_and_flow_nonnegative(normal_cols):
    for i in range(1, 7):
        assert np.all(normal_cols[f"p{i}"] >= 0.0), f"p{i} went negative"
    assert np.all(normal_cols["flowmeter"] >= 0.0), "flowmeter went negative"


def test_pump_status_is_binary(normal_cols):
    for ch in ("cpa_status", "turbo_status"):
        vals = np.unique(normal_cols[ch])
        assert set(vals.tolist()).issubset({0, 1}), f"{ch} not a 0/1 status bit"


def test_magnet_channels_track_4k_stage(normal_cols, normal_sim_module):
    """temp6/temp7 (magnet sensors) sit just above the 4K flange (temp2)."""
    four_k_end = normal_sim_module["T"][-1, 1]
    for ch in ("temp6_T", "temp7_T"):
        end = float(normal_cols[ch][-1])
        # within noise + magnet offset of the 4K stage, and above it
        assert end == pytest.approx(four_k_end, rel=0.1, abs=C.MAGNET_OFFSET_K + 0.5)


def test_magnet_quench_raises_magnet_telemetry(normal_cols):
    """A magnet_quench must drive the magnet sensor channels (temp6/temp7) up."""
    spec = make_spec("magnet_quench", target_stage=1)  # FOURK
    sim = run_sim(spec)
    quench_cols = emit(sim, spec, seed=1)
    for ch in ("temp6_T", "temp7_T"):
        assert quench_cols[ch][-1] > normal_cols[ch][-1], (
            f"magnet_quench did not raise {ch}"
        )


def test_emit_is_deterministic_for_fixed_seed(normal_sim_module):
    spec = FaultSpec("normal", onset_s=DURATION_S * 2, severity=0.0, target_stage=4)
    a = emit(normal_sim_module, spec, seed=7)
    b = emit(normal_sim_module, spec, seed=7)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"emit() not deterministic for channel {k}"
