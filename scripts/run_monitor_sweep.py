"""
run_monitor_sweep.py — the multi-seed / multi-fault continuous-monitoring study (live).

Runs the real-Sentinel continuous monitor (onnesim.continuous_monitor) across a grid of
faults x seeds and aggregates detection-latency distributions + confidence-gate trade-off
curves. This is the large-n replacement for the paper's single-run monitoring section.

Every poll is a live agent turn, so this costs API calls. Default grid is 5 faults x 5
seeds = 25 continuous 24 h runs (~41 polls each -> ~1000 live Sentinel turns).

Usage:
    .venv/bin/python scripts/run_monitor_sweep.py                 # full 5x5 grid
    .venv/bin/python scripts/run_monitor_sweep.py --faults helium_leak,magnet_quench --seeds 7,11
    .venv/bin/python scripts/run_monitor_sweep.py --reaggregate   # rebuild summary from disk
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import monitor_sweep as MS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", default=",".join(MS.DEFAULT_FAULTS),
                    help="comma-separated fault classes")
    ap.add_argument("--seeds", default="7,11,23,42,101", help="comma-separated integer seeds")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--poll-every-min", type=float, default=30.0)
    ap.add_argument("--backend", default="litellm")
    ap.add_argument("--workers", type=int, default=1, help="parallel cells (each cell's polls stay serial)")
    ap.add_argument("--out-dir", default="outputs/monitor_sweep")
    ap.add_argument("--out", default="outputs/monitor_sweep.json")
    ap.add_argument("--reaggregate", action="store_true",
                    help="skip running; rebuild the summary from per-cell JSONs on disk")
    args = ap.parse_args()

    cfg = MS.SweepConfig(
        faults=tuple(f.strip() for f in args.faults.split(",") if f.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        severity=args.severity, hours=args.hours, poll_every_min=args.poll_every_min,
        backend=args.backend, workers=args.workers, out_dir=args.out_dir, out_path=args.out)

    if args.reaggregate:
        summ = MS.reaggregate_from_dir(cfg.out_dir, cfg)
        with open(cfg.out_path, "w") as f:
            json.dump(summ, f, indent=2)
        print(f"[sweep] reaggregated {summ['n_cells']} cells -> {cfg.out_path}")
    else:
        summ = MS.run(cfg)

    print("\n" + "=" * 72)
    print("CONTINUOUS-MONITOR SWEEP SUMMARY")
    print("=" * 72)
    o = summ["overall"]
    lat = o["detection_latency_min"]
    if lat:
        print(f"  overall detection latency (min): median {lat['median']} "
              f"[{lat['p25']}, {lat['p75']}] (n={lat['n']}), range [{lat['min']},{lat['max']}]")
    print(f"  overall detection rate (any gate): {o['detection_rate_any_gate']}")
    print(f"\n  per-fault detection rate + median latency:")
    for fc, d in summ["per_fault"].items():
        L = d["latency_min_any_gate"]
        med = f"{L['median']} min" if L else "n/a (never detected)"
        print(f"    {fc:22s} rate={d['detection_rate']:.2f}  latency median={med}")
    print(f"\n  confidence-gate trade-off (pooled over all {summ['n_cells']} runs):")
    print(f"    {'gate':6s} {'FA/run':>8s} {'det.rate':>9s} {'lat.median':>11s} {'precision':>10s}")
    for g, d in summ["gate_curve"].items():
        fa = d["false_alarms_per_run"]
        L = d["latency_min"]
        print(f"    {g:6s} {fa['mean'] if fa else 0:>8} {d['detection_rate']:>9.2f} "
              f"{(L['median'] if L else '-'):>11} {str(d['mean_precision']):>10}")
    print("=" * 72)


if __name__ == "__main__":
    main()
