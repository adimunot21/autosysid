"""Logging setup: rich console output plus an optional plain-text file kept with each run."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

LOGGER_NAME = "autosysid"


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> logging.Logger:
    """Configure the package logger. Safe to call repeatedly (handlers are replaced, not stacked)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    logger.addHandler(RichHandler(show_path=False, rich_tracebacks=True, markup=False))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


if __name__ == "__main__":
    setup_logging(verbose=True)
    get_logger("selftest").info("logging works")
