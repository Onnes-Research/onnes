"""
OnnesSim fault injection — the piece that turns a simulator into a benchmark.

Each fault is a callable factory returning a `fault_hook(t_s, T) -> (extra_load, cooling_scale)`
consumed by model.simulate(). Faults have an onset time and a severity in [0,1].

⚠️ The *mechanisms* below are qualitatively motivated by how real dilution
fridges fail, but the magnitudes are PLACEHOLDER. The point of v0 is the
labeling + API shape, not physical fidelity. Realism audit by a domain
expert is a required later step (see PHYSICS_NOTES.md).

Fault taxonomy (label space):
    normal              — no fault
    blocked_impedance   — flow restriction; circulation drops, mxc warms
    thermal_touch       — a short between stages; extra conduction load
    helium_leak         — slow loss of mixture; cooling power fades over time
    pulse_tube_degrade  — upper-stage cooling weakens; whole chain drifts up
    heat_load_spike     — sudden parasitic load on a cold stage
    sensor_fault        — telemetry-only fault (handled in telemetry.py)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from . import constants as C

FAULT_CLASSES = [
    "normal",
    "blocked_impedance",
    "thermal_touch",
    "helium_leak",
    "pulse_tube_degrade",
    "heat_load_spike",
    "sensor_fault",
    "magnet_quench",     # 9T magnet on 4K flange self-heats & dumps load (arXiv:2602.05160)
]

# Stage index helpers
FIFTYK = C.STAGE_NAMES.index("50K")
FOURK = C.STAGE_NAMES.index("4K")     # magnet is thermalized to this flange
STILL = C.STAGE_NAMES.index("still")
COLDPLATE = C.STAGE_NAMES.index("coldplate")
MXC = C.STAGE_NAMES.index("mxc")


@dataclass
class FaultSpec:
    """Metadata describing an injected fault — this is the ground-truth label."""
    fault_class: str
    onset_s: float
    severity: float           # 0..1
    target_stage: int = -1    # -1 = whole system / not stage-specific
    flow_scale: float = 1.0   # multiplies reported circulation flow (telemetry)


def _ramp(t_s: float, onset_s: float, width_s: float = 1800.0) -> float:
    """Smooth 0->1 activation after onset."""
    if t_s <= onset_s:
        return 0.0
    return float(1.0 - np.exp(-(t_s - onset_s) / width_s))


def make_hook(spec: FaultSpec) -> Callable[[float, np.ndarray], tuple]:
    """Build the (extra_load, cooling_scale) hook for a given fault spec."""
    n = C.N_STAGES
    sev = float(np.clip(spec.severity, 0.0, 1.0))

    def hook(t_s: float, T: np.ndarray):
        extra = np.zeros(n)
        scale = np.ones(n)
        a = _ramp(t_s, spec.onset_s)

        fc = spec.fault_class
        if fc == "normal" or fc == "sensor_fault":
            pass  # no thermodynamic effect

        elif fc == "blocked_impedance":
            # Circulation restricted -> dilution cooling at still/mxc fades.
            scale[STILL] *= (1.0 - 0.8 * sev * a)
            scale[MXC] *= (1.0 - 0.9 * sev * a)

        elif fc == "thermal_touch":
            # Extra conductive bridge dumps heat onto a cold stage.
            stage = spec.target_stage if spec.target_stage >= 0 else MXC
            extra[stage] += 5.0e-4 * sev * a  # PLACEHOLDER [W]

        elif fc == "helium_leak":
            # Mixture slowly lost -> cooling power decays with time after onset.
            decay = 1.0 - 0.7 * sev * a
            scale[STILL] *= decay
            scale[COLDPLATE] *= decay
            scale[MXC] *= decay

        elif fc == "pulse_tube_degrade":
            # Upper-stage cooling weakens -> warms whole chain from the top.
            scale[0] *= (1.0 - 0.6 * sev * a)
            scale[1] *= (1.0 - 0.6 * sev * a)

        elif fc == "heat_load_spike":
            stage = spec.target_stage if spec.target_stage >= 0 else COLDPLATE
            extra[stage] += 2.0e-3 * sev * a  # PLACEHOLDER [W]

        elif fc == "magnet_quench":
            # 9T solenoid is thermalized to the 4K flange. A quench self-heats the
            # magnet and dumps a large transient load onto the 4K stage, which then
            # propagates down the chain. Magnet temp itself is handled in telemetry.
            extra[FOURK] += C.MAGNET_QUENCH_LOAD_W * sev * a  # PLACEHOLDER [W]

        return extra, scale

    return hook


def sample_fault(rng: np.random.Generator, duration_s: float) -> FaultSpec:
    """Draw a random labeled fault for dataset generation."""
    fc = rng.choice(FAULT_CLASSES)
    # Onset in the middle 60% of the run so there's pre- and post-fault signal.
    onset = float(rng.uniform(0.25, 0.75) * duration_s)
    sev = float(rng.uniform(0.3, 1.0))
    stage = int(rng.choice([COLDPLATE, MXC]))
    flow_scale = 1.0
    if fc == "blocked_impedance":
        flow_scale = float(1.0 - 0.6 * sev)   # visible in flow telemetry
    if fc == "magnet_quench":
        stage = FOURK                          # magnet dumps onto the 4K flange
    if fc == "normal":
        onset, sev = duration_s * 2, 0.0      # onset beyond end == never fires
    return FaultSpec(fault_class=fc, onset_s=onset, severity=sev,
                     target_stage=stage, flow_scale=flow_scale)
