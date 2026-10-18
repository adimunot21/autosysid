"""Exception hierarchy. The CLI catches `AutoSysIDError` and prints a one-line message instead of a traceback."""

from __future__ import annotations


class AutoSysIDError(Exception):
    """Base class for all expected, user-facing failures."""


class ConfigError(AutoSysIDError):
    """A configuration value is out of range or inconsistent."""


class HardwareError(AutoSysIDError):
    """The servo bus is missing, unresponsive, or returned malformed data."""


class Aborted(AutoSysIDError):
    """The operator declined to continue; nothing was moved."""


class SafetyViolation(HardwareError):
    """A safety watchdog tripped; the run was (or must be) stopped."""


class DataError(AutoSysIDError):
    """A recorded log or model file does not match the contract in docs/data-contract.md."""


class FitError(AutoSysIDError):
    """The optimization finished but the result is not trustworthy enough to export."""


class ExportError(AutoSysIDError):
    """The fitted values cannot be written into the MJCF unambiguously, or the written file failed verification."""
