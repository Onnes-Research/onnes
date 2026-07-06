"""OnnesSim test suite.

Tests exercise the public API only (onnesim.simulate / faults / telemetry.emit /
evaluate.score). They do NOT depend on the PLACEHOLDER magnitudes in
constants.py — assertions check structural / relational invariants (ordering,
finiteness, fault raises the right stage vs a matched baseline, schema columns,
scoring math) so they stay valid when M2 replaces the constants with validated
values.
"""
