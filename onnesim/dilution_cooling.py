"""
dilution_cooling.py — real 3He/4He dilution-unit cooling power (T^2 law).

This replaces OnnesSim's phenomenological LINEAR cooling at the cold stages with the
physically-correct dilution-cooling relation, which was the #1 credibility threat
flagged in docs/PHYSICS_REVIEW.md ("linear G*dT erases the base-temperature floor").

Physics (standard, e.g. Pobell "Matter and Methods at Low Temperatures"):
    Q_MXC(T) = n_dot_3 * (95 * T_MXC^2 - 11 * T_in^2),   T_in = f * T_MXC
    => Q_MXC(T) = n_dot_3 * (95 - 11 f^2) * T_MXC^2
    => T_MXC(Q_load) = sqrt( Q_load / [ n_dot_3 (95 - 11 f^2) ] )
Cooling VANISHES as T->0 (unlike linear cooling), so there is a real base-temperature
floor set by the parasitic load — the physically essential behavior.

Attribution: the specific coefficient form (95/11), T_in-factor parameterization, and
the LD400 calibration (n_dot ~ 500 umol/s, f = 1.5, 14 uW -> 20 mK) follow the
MIT-licensed implementation `paulggin/cryostat-thermal-model` (experiments/
dilution_unit.py, MIT (c) 2026 Paul Gin) and standard dilution-fridge theory.
Re-implemented here (not copied) with attribution.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

H_DILUTE = 95.0        # J/(mol K^2), dilute-phase enthalpy coefficient
H_CONC = 11.0          # J/(mol K^2), concentrated-phase enthalpy coefficient


@dataclass
class DilutionUnit:
    n_dot_3: float = 500e-6      # 3He circulation rate [mol/s]  (~500 umol/s)
    T_in_factor: float = 1.5     # heat-exchanger inlet ratio T_in / T_MXC
    Q_cp_100mK_W: float = 300e-6  # cold-plate cooling power at 100 mK [W]
    Q_still_nominal_W: float = 30e-3  # still cooling power [W]

    def Q_MXC(self, T_MXC_K: float) -> float:
        """Mixing-chamber cooling power [W] at temperature T_MXC_K."""
        T_in = self.T_in_factor * T_MXC_K
        return self.n_dot_3 * max(H_DILUTE * T_MXC_K**2 - H_CONC * T_in**2, 0.0)

    def T_MXC_at_load(self, Q_load_W: float) -> float:
        """Steady MXC temperature [K] for a given parasitic heat load [W].
        This is the base-temperature-floor relation: T ~ sqrt(Q)."""
        coef = self.n_dot_3 * (H_DILUTE - H_CONC * self.T_in_factor**2)
        if coef <= 0:
            raise ValueError("Non-positive dilution coefficient; check T_in_factor.")
        return float(np.sqrt(max(Q_load_W, 0.0) / coef))

    def T_cp_at_load(self, Q_load_W: float) -> float:
        """Cold-plate temperature [K] for a load; anchored to LD400 100 mK spec, T^2."""
        return 0.1 * float(np.sqrt(max(Q_load_W, 0.0) / self.Q_cp_100mK_W))

    def T_still_at_load(self, Q_load_W: float) -> float:
        """Still temperature [K] for a load; linear anchor at 0.8 K nominal."""
        return 0.8 * (Q_load_W / max(self.Q_still_nominal_W, 1e-12))
