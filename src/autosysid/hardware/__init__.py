"""Hardware layer: servo bus abstraction, calibration, safety, trajectories and the fixed-rate recorder.

Everything above `bus.ServoBus` is hardware-agnostic and is tested against `MockBus`; only `feetech.FeetechBus` talks
to a real serial port.
"""
