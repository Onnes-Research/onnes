"""
OnnesSim thermal model — phenomenological lumped-stage dynamics.

⚠️ PHENOMENOLOGICAL, NOT FIRST-PRINCIPLES. See PHYSICS_NOTES.md.

Each stage i has a temperature T_i. We integrate a simple energy-balance ODE:

    C_i dT_i/dt = Q_load_i(t)                 (parasitic + fault heat in)
                 + Q_couple_from_warmer        (conduction down from i-1)
                 - Q_couple_to_colder          (conduction to i+1)
                 - Q_cool_i(t)                  (available cooling, ramped by pulse tube)

The pulse tube ramps cooling in over PULSE_TUBE_TAU so the upper stages
pre-cool first, loosely reproducing the multi-day 300K -> mK sequence.

The state vector is the 5 stage temperatures. Physics is written with numpy
array ops over stages (no python loop over stages) so a later torch port can
add a leading batch dimension for fleet-scale simulation on GPU.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from . import constants as C


@dataclass
class FridgeParams:
    """All per-stage parameters. Defaults come from constants.py (PLACEHOLDER)."""
    t_target: np.ndarray = field(default_factory=lambda: C.T_TARGET.copy())
    heat_capacity: np.ndarray = field(default_factory=lambda: C.HEAT_CAPACITY.copy())
    cooling_power: np.ndarray = field(default_factory=lambda: C.COOLING_POWER.copy())
    static_load: np.ndarray = field(default_factory=lambda: C.STATIC_LOAD.copy())
    coupling: np.ndarray = field(default_factory=lambda: C.COUPLING.copy())
    pulse_tube_tau: float = C.PULSE_TUBE_TAU
    nominal_flow: float = C.NOMINAL_FLOW

    def perturbed(self, rng: np.random.Generator, rel: float = 0.15) -> "FridgeParams":
        """Return a copy with multiplicative jitter — one 'virtual fridge' in a fleet."""
        def j(x):
            return x * (1.0 + rel * rng.standard_normal(np.shape(x)))
        return FridgeParams(
            t_target=self.t_target.copy(),
            heat_capacity=np.abs(j(self.heat_capacity)),
            cooling_power=np.abs(j(self.cooling_power)),
            static_load=np.abs(j(self.static_load)),
            coupling=np.abs(j(self.coupling)),
            pulse_tube_tau=float(abs(j(self.pulse_tube_tau))),
            nominal_flow=float(abs(j(self.nominal_flow))),
        )


def pulse_tube_factor(t_s: float, tau: float) -> float:
    """Cooling availability in [0,1], ramping up as the pulse tube engages."""
    return 1.0 - np.exp(-t_s / tau)


def derivatives(
    T: np.ndarray,
    t_s: float,
    params: FridgeParams,
    extra_load: np.ndarray,
    cooling_scale: np.ndarray,
) -> np.ndarray:
    """dT/dt for all stages. `extra_load` [W] and `cooling_scale` [-] are the
    fault hooks (per-stage). Vectorized over stages.

    Cooling model (v0, phenomenological): each stage sheds heat via a cooling
    conductance toward its target, q_cool_i = g_i * ramp * gate_i * (T_i-target_i)_+.
    `gate_i` makes a cold stage's cooling switch on only once the warmer stage
    above it is near its own target — this reproduces the *sequential* top-down
    cooldown of a real fridge instead of every stage cooling at once.
    """
    n = C.N_STAGES
    ramp = pulse_tube_factor(t_s, params.pulse_tube_tau)

    # Conduction down the chain: flux_i = k_i * (T_i - T_{i+1}) for i=0..n-2
    dT_pairs = T[:-1] - T[1:]
    flux_down = params.coupling[:-1] * dT_pairs           # length n-1, >0 warm->cold

    q_from_warmer = np.zeros(n)
    q_from_warmer[1:] = flux_down                          # stage i gains from i-1
    q_to_colder = np.zeros(n)
    q_to_colder[:-1] = flux_down                           # stage i loses to i+1

    # Sequential gate: stage i (i>=1) cools once the warmer stage i-1 nears target.
    gate = np.ones(n)
    warm_ratio = T[:-1] / np.maximum(C.GATE_FACTOR * params.t_target[:-1], 1e-9)
    gate[1:] = 1.0 / (1.0 + warm_ratio ** C.GATE_SHARP)

    temp_err = np.maximum(T - params.t_target, 0.0)
    q_cool = C.G_COOL * ramp * gate * cooling_scale * temp_err

    q_in = params.static_load + extra_load + q_from_warmer
    q_out = q_to_colder + q_cool

    return (q_in - q_out) / params.heat_capacity


def simulate(
    params: Optional[FridgeParams] = None,
    duration_s: float = 36.0 * 3600.0,
    dt_s: float = 30.0,
    fault_hook: Optional[Callable[[float, np.ndarray], tuple]] = None,
    T0: Optional[np.ndarray] = None,
    seed: int = 0,
) -> dict:
    """
    Integrate the fridge with explicit RK4.

    fault_hook(t_s, T) -> (extra_load[n], cooling_scale[n]) lets a fault modify
    heat loads and cooling at each step. Default: no fault.

    Returns dict with time [s] and temperature history [n_steps, n_stages].
    """
    if params is None:
        params = FridgeParams()
    if T0 is None:
        T0 = np.full(C.N_STAGES, C.T_ROOM, dtype=float)

    n_steps = int(duration_s / dt_s) + 1
    ts = np.arange(n_steps) * dt_s
    T_hist = np.empty((n_steps, C.N_STAGES), dtype=float)
    T = T0.astype(float).copy()
    T_hist[0] = T

    no_load = np.zeros(C.N_STAGES)
    unit_scale = np.ones(C.N_STAGES)

    def rhs(Tx, tx):
        if fault_hook is not None:
            extra, scale = fault_hook(tx, Tx)
        else:
            extra, scale = no_load, unit_scale
        return derivatives(Tx, tx, params, extra, scale)

    for k in range(1, n_steps):
        t = ts[k - 1]
        k1 = rhs(T, t)
        k2 = rhs(T + 0.5 * dt_s * k1, t + 0.5 * dt_s)
        k3 = rhs(T + 0.5 * dt_s * k2, t + 0.5 * dt_s)
        k4 = rhs(T + dt_s * k3, t + dt_s)
        T = T + (dt_s / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Physical floor: nothing goes below ~base target; avoids negative T from stiff steps.
        T = np.maximum(T, 0.5 * params.t_target)
        T_hist[k] = T

    return {"t_s": ts, "T": T_hist, "stages": C.STAGE_NAMES}
