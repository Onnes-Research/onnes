"""OnnesSim — phenomenological dilution-refrigerator simulator for cryo-ops ML research.

⚠️ v0 research prototype. Physics is phenomenological and constants are PLACEHOLDER.
See PHYSICS_NOTES.md before using any output for a scientific claim.
"""
from .model import FridgeParams, simulate
from .faults import FaultSpec, make_hook, sample_fault, FAULT_CLASSES
from .telemetry import emit, to_csv

__version__ = "0.0.1"
__all__ = [
    "FridgeParams", "simulate",
    "FaultSpec", "make_hook", "sample_fault", "FAULT_CLASSES",
    "emit", "to_csv",
]
