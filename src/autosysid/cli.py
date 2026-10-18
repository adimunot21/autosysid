"""Command-line interface. Commands are added phase by phase; `info` verifies the install."""

from __future__ import annotations

import importlib.metadata
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from autosysid import __version__
from autosysid.errors import AutoSysIDError
from autosysid.logging_utils import setup_logging

app = typer.Typer(help="Calibrate a MuJoCo model of an SO-101 arm from real hardware data.", no_args_is_help=True)
console = Console()


def _version_callback(show: bool) -> None:
    if show:
        console.print(f"autosysid {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_version_callback, is_eager=True)] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    setup_logging(verbose=verbose)


@app.command()
def info() -> None:
    """Show versions of the key dependencies and whether mujoco.sysid is usable."""
    table = Table("component", "version / status")
    table.add_row("autosysid", __version__)
    table.add_row("python", sys.version.split()[0])
    for package in ("mujoco", "numpy", "scipy", "feetech-servo-sdk"):
        try:
            table.add_row(package, importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            table.add_row(package, "[red]NOT INSTALLED[/red]")

    try:
        from mujoco import sysid

        table.add_row("mujoco.sysid", "ok" if hasattr(sysid, "optimize") else "[red]extra missing: mujoco[sysid][/red]")
    except ImportError as error:
        table.add_row("mujoco.sysid", f"[red]unavailable: {error}[/red]")
    console.print(table)


def run() -> None:
    """Entry point wrapper: expected failures print one line, unexpected ones keep their traceback."""
    try:
        app()
    except AutoSysIDError as error:
        console.print(f"[red]error:[/red] {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    run()
