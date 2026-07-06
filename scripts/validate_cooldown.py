"""
validate_cooldown.py — fit OnnesSim against a REFERENCE cooldown curve and report
reproduction error. This is the M2 validation harness: it converts "not validated"
into a cited number the moment a real digitized curve is dropped in.

USAGE:
    # with a real digitized curve (preferred):
    python scripts/validate_cooldown.py --ref data/real/cooldown_ref.csv
    # reference CSV columns:  stage,time_h,temp_K
    #   stage in {4K, mxc, ...} matching onnesim stage names; one row per point.

    # illustrative demo (NO real data — clearly labeled):
    python scripts/validate_cooldown.py --demo

⚠️ HONESTY: with --demo, the "reference" is a SINGLE published TIMING point, not a
digitized curve: the condensation-driven DR paper (J. Appl. Phys. 135, 235901,
2024) reports ~300 K -> 4 K in ~30 h. That is a coarse timescale sanity check ONLY.
A real validation needs a digitized multi-point curve (see docs/REAL_DATA_OPTIONS.md,
best candidate arXiv:2501.04471). Do not report --demo output as "validated."
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import FridgeParams, simulate
from onnesim import constants as C


def load_ref(path: str) -> list:
    """Load reference points: list of (stage_name, time_s, temp_K)."""
    pts = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            pts.append((row["stage"].strip(),
                        float(row["time_h"]) * 3600.0,
                        float(row["temp_K"])))
    return pts


def demo_ref() -> list:
    """ILLUSTRATIVE ONLY — single published timing, not a digitized curve.
    J. Appl. Phys. 135, 235901 (2024): ~300K -> 4K in ~30 h (4K stage)."""
    return [("4K", 30.0 * 3600.0, 4.0)]


def sim_stage_temp_at(tau_s: float, stage: str, t_target_s: float) -> float:
    """Run OnnesSim with a given PULSE_TUBE_TAU and read one stage temp at time t."""
    p = FridgeParams()
    p.pulse_tube_tau = tau_s
    dur = max(t_target_s * 1.5, 12 * 3600.0)
    sim = simulate(p, duration_s=dur, dt_s=120.0)
    idx = int(np.searchsorted(sim["t_s"], t_target_s))
    idx = min(idx, len(sim["t_s"]) - 1)
    si = C.STAGE_NAMES.index(stage)
    return float(sim["T"][idx, si])


def rms_error(tau_s: float, ref: list) -> float:
    """RMS relative error of sim vs reference points at a given tau."""
    errs = []
    for stage, t_s, temp in ref:
        got = sim_stage_temp_at(tau_s, stage, t_s)
        errs.append((got - temp) / (abs(temp) + 1e-9))
    return float(np.sqrt(np.mean(np.square(errs))))


def fit_tau(ref: list, grid=None) -> tuple:
    """Coarse 1-D search over PULSE_TUBE_TAU to minimize RMS error."""
    if grid is None:
        grid = np.linspace(1.0, 15.0, 15) * 3600.0  # 1..15 h
    best_tau, best_err = None, np.inf
    curve = []
    for tau in grid:
        e = rms_error(tau, ref)
        curve.append((tau / 3600.0, e))
        if e < best_err:
            best_err, best_tau = e, tau
    return best_tau, best_err, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="CSV of real reference points (stage,time_h,temp_K)")
    ap.add_argument("--demo", action="store_true",
                    help="use the ILLUSTRATIVE single published timing (not real validation)")
    args = ap.parse_args()

    if args.ref:
        ref = load_ref(args.ref)
        source = f"real reference: {args.ref}"
        illustrative = False
    elif args.demo:
        ref = demo_ref()
        source = "ILLUSTRATIVE demo (J.Appl.Phys.135,235901 timing only)"
        illustrative = True
    else:
        print("Provide --ref <csv> (real curve) or --demo (illustrative). "
              "See docs/REAL_DATA_OPTIONS.md.")
        sys.exit(1)

    print(f"[validate] reference: {source}  ({len(ref)} point(s))")
    best_tau, best_err, curve = fit_tau(ref)
    print(f"[validate] current constants.PULSE_TUBE_TAU = {C.PULSE_TUBE_TAU/3600:.1f} h")
    print(f"[validate] best-fit PULSE_TUBE_TAU = {best_tau/3600:.2f} h  "
          f"(RMS rel. error {best_err*100:.1f}%)")
    print("[validate] tau[h] -> RMS error:")
    for tau_h, e in curve:
        mark = "  <-- best" if abs(tau_h - best_tau/3600) < 1e-6 else ""
        print(f"    {tau_h:5.1f}   {e*100:6.1f}%{mark}")

    if illustrative:
        print("\n⚠️  DEMO ONLY — a single timing point is NOT validation. Replace with a")
        print("   digitized multi-point curve (docs/REAL_DATA_OPTIONS.md) before any claim.")


if __name__ == "__main__":
    main()
