"""
OnnesSim constants — dilution-refrigerator stage parameters.

=============================================================================
 ⚠️  EVERY NUMBER IN THIS FILE IS A **PLACEHOLDER**.
 They are chosen to be order-of-magnitude *plausible* for a dry dilution
 refrigerator (Bluefors LD400-class), NOT measured or validated.
 Do not treat any output as quantitatively correct until these are
 replaced with values validated against real data or by a domain expert.
 See PHYSICS_NOTES.md.
=============================================================================

A dry dilution fridge has ~5 temperature stages, from warm to cold:
    50K plate  -> 4K plate  -> still (~0.7 K) -> cold plate (~50 mK) -> mixing chamber (~10 mK)
Cooling is provided by a pulse tube (upper stages) and the He-3/He-4
dilution process (lower stages).
"""

from __future__ import annotations
import numpy as np

# Stage names, warm -> cold. Index order is used everywhere.
STAGE_NAMES = ["50K", "4K", "still", "coldplate", "mxc"]
N_STAGES = len(STAGE_NAMES)

# Target (steady-state) temperatures per stage [K].
# CALIBRATED against CryowalaCore param_functions.py `fridge_ours` =
#   [46, 3.94, 1.227, 0.150, 0.020] K and `fridge_100q` = [35, 2.85, 0.882, 0.082, 0.006] K
# (github.com/CirQuS-UTS/CryowalaCore), Bluefors LD (base < 10 mK), PT415 (4K base 2.8 K).
# coldplate raised 50->100 mK per review (50 mK was below the CP/still-plate range).
T_TARGET = np.array([45.0, 3.8, 0.70, 0.100, 0.010], dtype=float)

# Room / start temperature [K].
T_ROOM = 300.0

# Effective heat capacity per stage [J/K]. PLACEHOLDER.
# Warm stages are massive (large C); cold stages are light but C collapses
# with temperature in reality — here folded into a constant placeholder.
HEAT_CAPACITY = np.array([4.0e3, 8.0e2, 5.0e1, 2.0e0, 5.0e-1], dtype=float)

# Cooling power available at each stage [W].
# CALIBRATED against verified primary sources (see docs/PHYSICS_REVIEW.md):
#   50K, 4K  -> Bluefors PT415: 40 W @ 45 K, 1.5 W @ 4.2 K (exact match).
#   still, coldplate, mxc -> CryowalaCore param_functions.py `fridge_ours`:
#       cool_p = [10, 0.5, 30e-3, 300e-6, 20e-6] W  (github.com/CirQuS-UTS/CryowalaCore).
# NOTE: real mxc cooling is strongly T-dependent (~n3*T^2, vanishes as T->0); this
# flat value is the ~20mK operating point only. Full T^2 law is roadmap M3.
# In v0 these are documentation/caps; dynamics use the G_COOL conductance below.
COOLING_POWER = np.array([40.0, 1.5, 3.0e-2, 3.0e-4, 2.0e-5], dtype=float)

# Cooling CONDUCTANCE per stage [W/K]. PLACEHOLDER.
# v0 models cooling as g_i * (T_i - target_i)_+ , i.e. each stage relaxes toward
# its target with time constant tau_i = C_i / g_i once the pulse tube engages.
# Chosen so steady-state offset (load / g) is small at every stage.
G_COOL = np.array([8.0, 2.0, 5.0e-2, 1.0e-2, 8.0e-3], dtype=float)

# Static parasitic heat load onto each stage [W] (radiation, conduction,
# wiring). This is exactly what Cryowala models properly.
# FIX (docs/PHYSICS_REVIEW.md Threat #1): mxc load was 15 µW ≈ the ENTIRE
# ~14-20 µW mxc cooling budget (Bluefors LD / Cryowala) -> a real fridge could
# not reach base. Reduced to ~0.5 µW so load << budget. Upper stages plausible
# order-of-magnitude (radiation/wiring), still UNVERIFIED magnitudes.
STATIC_LOAD = np.array([15.0, 0.30, 8.0e-4, 3.0e-5, 5.0e-7], dtype=float)

# Inter-stage thermal conductance to the next-colder stage [W/K]. PLACEHOLDER.
# Index i couples stage i to stage i+1; last entry unused. Kept small: real
# fridge stages are deliberately well isolated, so coupling must NOT dominate
# each stage's own cooling conductance (else all stages drag to a common T).
COUPLING = np.array([1.0e-2, 5.0e-3, 2.0e-4, 2.0e-5, 0.0], dtype=float)

# Pulse-tube time constant [s] — sets how fast upper stages pre-cool. PLACEHOLDER.
PULSE_TUBE_TAU = 6.0 * 3600.0

# Sequential-cooldown gating (dimensionless). PLACEHOLDER, tuning knobs only.
# A cold stage's cooling engages once the warmer stage above it drops to within
# ~GATE_FACTOR of that warmer stage's own target; GATE_SHARP sets how sharp the
# handover is. This yields the characteristic top-down staged cooldown.
GATE_FACTOR = 3.0
GATE_SHARP = 4.0

# Nominal helium circulation flow rate [mmol/s], reported in telemetry. PLACEHOLDER.
NOMINAL_FLOW = 0.80

# Nominal still/condensing-line pressures P1..P6 [mbar]. PLACEHOLDER.
# Loosely mimics a Bluefors gas-handling schema (condense, still, tank, etc.).
NOMINAL_PRESSURE = np.array([1.5e3, 0.35, 8.0, 1.0e3, 2.0e-2, 5.0e2], dtype=float)

# Sensor Gaussian noise (relative std) applied by the telemetry emitter. PLACEHOLDER.
TEMP_NOISE_REL = 0.010
PRESSURE_NOISE_REL = 0.015
FLOW_NOISE_REL = 0.020

# --- Magnet + 8-channel telemetry -------------------------------------------
# VERIFIED schema source: FRTMS, arXiv:2602.05160 (Esposito, Greenstein, Speller)
# and the repo Speller-Laboratory/JHUFridgeMonitoring-public. The JHU LD400 has
# a 9 T solenoid thermalized to the 4 K flange, with two temperature sensors in
# the magnet and spare empty readout channels; DB tables are temp1..temp8.
# We model the magnet at the TELEMETRY layer (not as an ODE node): its baseline
# tracks the 4 K stage; a quench makes it self-heat and dumps heat to the flange.
MAGNET_OFFSET_K = 0.05          # PLACEHOLDER: magnet sits slightly above 4K flange
MAGNET_QUENCH_SPIKE_K = 8.0     # PLACEHOLDER: magnet temp excursion during a quench
MAGNET_QUENCH_LOAD_W = 2.0      # PLACEHOLDER: heat dumped onto the 4K flange in a quench
SPARE_CHANNEL_K = 300.0         # empty readout channel (real ones rail/NaN); PLACEHOLDER
