"""Command-line interface.

autosysid doctor    read-only health check of the bus (no motion, no writes)
autosysid record    run the excitation plan on the arm and save raw logs
autosysid fit       identify dynamics + latency from a run directory (no hardware needed)
autosysid validate  score the fitted model on held-out motion
autosysid export    write the calibrated MJCF + calibration.json
autosysid calibrate record -> fit -> validate -> export in one go
autosysid demo      the whole pipeline on a simulated arm (no hardware)
"""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from autosysid import __version__, pipeline
from autosysid.config import ARM_JOINT_NAMES, FitConfig, RecordConfig, SafetyLimits
from autosysid.errors import AutoSysIDError
from autosysid.exporter import export_calibrated
from autosysid.logging_utils import setup_logging
from autosysid.mock import DEMO_DELAY_S, DEMO_TRUTH_SCALE, MINI_ARM_XML, MockArm

app = typer.Typer(help="Calibrate a MuJoCo model of an SO-101 arm from real hardware data.", no_args_is_help=True)
console = Console()
DEFAULT_PORT = RecordConfig().port

XmlOption = Annotated[
    Path, typer.Option("--xml", help="MJCF to calibrate (e.g. scene.xml of the SO-ARM100 SO101 model).")
]
PortOption = Annotated[str, typer.Option(help="Serial port of the arm.")]
CalibrationOption = Annotated[
    Path | None, typer.Option(help="LeRobot calibration JSON (default: the SO-101 follower file in the HF cache).")
]
MockOption = Annotated[bool, typer.Option("--mock", help="Use a simulated arm instead of hardware.")]
RunDirOption = Annotated[Path | None, typer.Option(help="Run directory (default: runs/<timestamp>).")]


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


def _run_dir(run_dir: Path | None, prefix: str = "") -> Path:
    path = run_dir or Path("runs") / f"{prefix}{time.strftime('%Y%m%d-%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _print_preflight(report) -> None:
    console.print(
        "[bold]Simulation pre-flight[/bold] (nominal model, the reduced torque limit, exactly what will be sent)"
    )
    console.print(report.summary())


def _print_fits(fits) -> None:
    table = Table(
        "joint",
        "parameter",
        "nominal",
        "fitted",
        "95% CI",
        "status",
        title="Identified parameters",
        caption="kp, kv, friction are weakly observable and may be off even when marked ok; trust held-out error.",
    )
    for fit in fits:
        values, nominal = fit.dynamics.to_dict(), fit.nominal.to_dict()
        for name in values:
            ci = fit.relative_ci.get(name, float("inf"))
            flagged = name in fit.unidentifiable
            table.add_row(
                fit.joint if name == "kp" else "",
                name,
                f"{nominal[name]:.4g}",
                f"{values[name]:.4g}",
                "n/a" if ci == float("inf") else f"+-{100 * ci:.0f}%",
                "[yellow]not identified: left at nominal[/yellow]" if flagged else "ok",
            )
        delay_ci = fit.relative_ci.get("delay", float("inf"))
        table.add_row(
            "",
            "delay (ms)",
            "-",
            f"{fit.delay_s * 1e3:.2f}",
            "n/a" if delay_ci == float("inf") else f"+-{delay_ci * 1e3:.2f} ms",
            "[yellow]not identified[/yellow]" if "delay" in fit.unidentifiable else "ok",
        )
        verdict = "[green]accepted[/green]" if fit.accepted else "[red]REJECTED (cost above noise-floor margin)[/red]"
        table.add_row("", "cost / floor", "-", f"{fit.cost / fit.noise_floor:.2f}x", "-", verdict)
    console.print(table)


def _print_validation(results) -> None:
    if not results:
        console.print("[yellow]no held-out data, so no validation (record with stage 3)[/yellow]")
        return
    table = Table("joint", "nominal model", "fitted model", "improvement", title="Held-out prediction error (RMS)")
    for r in results:
        table.add_row(
            r.joint,
            f"{r.rms_nominal_rad * 1e3:.2f} mrad",
            f"{r.rms_fitted_rad * 1e3:.2f} mrad",
            f"{r.improvement:.1f}x",
        )
    console.print(table)


def _record_config(bus, mock, port, baudrate, joints, amplitude_scale, torque_limit) -> RecordConfig:
    default = (bus.joints[0],) if mock else ARM_JOINT_NAMES
    return RecordConfig(
        port=port,
        baudrate=baudrate,
        joints=pipeline.parse_joints(joints, bus.joints) if joints else default,
        amplitude_scale=amplitude_scale,
        safety=SafetyLimits(torque_limit_fraction=torque_limit),
    )


def _confirm_factory(mock: bool, yes: bool, config: RecordConfig, stages: tuple[int, ...]):
    def confirm(report) -> bool:
        _print_preflight(report)
        if mock or yes:
            return True
        console.print(
            "\n[bold red]The arm is about to MOVE.[/bold red]\n"
            f"  joints: {', '.join(config.joints)}   stages: {stages}   amplitude scale: {config.amplitude_scale}\n"
            f"  torque limit will be set to {config.safety.torque_limit_fraction:.0%} of full "
            "(registers Torque_Limit and are restored afterwards)\n"
            "  Clear the workspace, keep a hand near the power switch. Ctrl-C stops safely."
        )
        return typer.confirm("Proceed?", default=False)

    return confirm


@app.command()
def doctor(
    port: PortOption = DEFAULT_PORT,
    calibration: CalibrationOption = None,
    baudrate: int = 1_000_000,
    mock: MockOption = False,
    xml: Annotated[Path, typer.Option("--xml")] = MINI_ARM_XML,
) -> None:
    """Read-only check: ping every servo, read pose, temperature, voltage, settings. Moves nothing, writes nothing."""
    bus, arm_calibration, _ = pipeline.open_bus(xml, mock, port, baudrate, calibration)
    try:
        snapshot = bus.read_state()
        radians = arm_calibration.ticks_to_rad(snapshot.position_ticks)
        lower, upper = arm_calibration.soft_limits_rad(SafetyLimits().soft_limit_fraction)
        torque, acceleration = bus.read_setting("torque_limit"), bus.read_setting("acceleration")
        table = Table("joint", "ticks", "rad", "in soft range", "temp C", "volt V", "torque limit", "accel", title=port)
        table.columns[0].no_wrap = True
        for i, name in enumerate(bus.joints):
            inside = lower[i] <= radians[i] <= upper[i]
            table.add_row(
                name,
                str(snapshot.position_ticks[i]),
                f"{radians[i]:+.3f}",
                "yes" if inside else "[red]NO: move it toward the middle[/red]",
                f"{snapshot.temperature_c[i]:.0f}",
                f"{snapshot.voltage_v[i]:.1f}",
                str(torque[i]),
                str(acceleration[i]),
            )
        console.print(table)
        console.print(f"sync read: {snapshot.read_duration_s * 1e3:.2f} ms for {len(bus.joints)} servos")
    finally:
        bus.close()


@app.command()
def record(
    xml: XmlOption,
    port: PortOption = DEFAULT_PORT,
    calibration: CalibrationOption = None,
    baudrate: int = 1_000_000,
    mock: MockOption = False,
    run_dir: RunDirOption = None,
    joints: Annotated[
        str | None, typer.Option(help="Comma-separated joints to excite (default: the 5 arm joints).")
    ] = None,
    stages: Annotated[str, typer.Option(help="Which stages to run, e.g. 1 or 1,2,3.")] = "1,2,3",
    amplitude_scale: Annotated[float, typer.Option(help="0-1 multiplier on all amplitudes and speeds.")] = 0.5,
    torque_limit: Annotated[float, typer.Option(help="Fraction of full servo torque while recording.")] = 0.7,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Run the excitation plan (pre-flight first) and save the raw logs to the run directory."""
    stage_tuple = pipeline.parse_stages(stages)
    bus, arm_calibration, clock = pipeline.open_bus(xml, mock, port, baudrate, calibration)
    try:
        config = _record_config(bus, mock, port, baudrate, joints, amplitude_scale, torque_limit)
        path = _run_dir(run_dir)
        with console.status("recording..."):
            logs = pipeline.record(
                bus,
                arm_calibration,
                clock,
                xml,
                config,
                stage_tuple,
                path,
                confirm=_confirm_factory(mock, yes, config, stage_tuple),
            )
        console.print(f"recorded {len(logs)} segments -> [bold]{path}[/bold]")
    finally:
        bus.close()


@app.command()
def fit(
    xml: XmlOption,
    run_dir: Annotated[Path, typer.Option(help="Run directory containing raw/ logs.")],
    joints: Annotated[str | None, typer.Option(help="Joints to fit (default: every joint with stage-2 logs).")] = None,
    restarts: Annotated[int, typer.Option(help="Extra perturbed starts per joint (more = more robust, slower).")] = 2,
) -> None:
    """Identify servo dynamics and latency from recorded logs. Needs no hardware."""
    logs = pipeline.load_logs(run_dir)
    chosen = pipeline.parse_joints(joints, pipeline.excited_joints(logs))
    with console.status(f"fitting {', '.join(chosen)} (about 1-2 min per joint)..."):
        fits = pipeline.fit_run(xml, logs, chosen, FitConfig(max_restarts=restarts))
    pipeline.save_fits(fits, run_dir / "fits.json")
    _print_fits(fits)


@app.command()
def validate(
    xml: XmlOption,
    run_dir: Annotated[Path, typer.Option(help="Run directory with raw/ logs and fits.json.")],
) -> None:
    """Score nominal vs fitted model on held-out (stage 3) motion that was never used for fitting."""
    logs, fits = pipeline.load_logs(run_dir), pipeline.load_fits(run_dir / "fits.json")
    results = pipeline.validate_run(xml, logs, fits, FitConfig())
    pipeline.save_validation(results, run_dir / "validation.json")
    _print_validation(results)


@app.command()
def export(
    xml: XmlOption,
    run_dir: Annotated[Path, typer.Option(help="Run directory with fits.json.")],
    out: Annotated[
        Path | None, typer.Option(help="Output MJCF (same folder as --xml; default <name>_calibrated.xml).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Also export fits that were not accepted.")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Write the calibrated MJCF (explicit per-joint values) and calibration.json next to it."""
    fits = pipeline.load_fits(run_dir / "fits.json")
    target = out or xml.with_name(f"{xml.stem}_calibrated{xml.suffix}")
    result = export_calibrated(xml, target, fits, FitConfig(), force=force, overwrite=overwrite)
    for item in result.joints:
        mark = "[green]exported[/green]" if item.exported else f"[yellow]skipped: {item.reason}[/yellow]"
        console.print(f"{item.joint}: {mark}")
    console.print(f"wrote {', '.join(str(p) for p in result.xml_paths)} and {result.calibration_path}")


@app.command()
def calibrate(
    xml: XmlOption,
    out: Annotated[Path | None, typer.Option(help="Output MJCF (default <name>_calibrated.xml next to --xml).")] = None,
    port: PortOption = DEFAULT_PORT,
    calibration: CalibrationOption = None,
    baudrate: int = 1_000_000,
    mock: MockOption = False,
    run_dir: RunDirOption = None,
    joints: Annotated[str | None, typer.Option(help="Comma-separated joints (default: the 5 arm joints).")] = None,
    stages: Annotated[str, typer.Option(help="Stages to record, e.g. 2,3.")] = "1,2,3",
    amplitude_scale: Annotated[float, typer.Option()] = 0.5,
    torque_limit: Annotated[float, typer.Option()] = 0.7,
    restarts: Annotated[int, typer.Option()] = 2,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """record -> fit -> validate -> export."""
    stage_tuple = pipeline.parse_stages(stages)
    bus, arm_calibration, clock = pipeline.open_bus(xml, mock, port, baudrate, calibration)
    try:
        config = _record_config(bus, mock, port, baudrate, joints, amplitude_scale, torque_limit)
        path = _run_dir(run_dir)
        logs = pipeline.record(
            bus,
            arm_calibration,
            clock,
            xml,
            config,
            stage_tuple,
            path,
            confirm=_confirm_factory(mock, yes, config, stage_tuple),
        )
    finally:
        bus.close()
    _fit_validate_export(xml, out, path, logs, config.joints, restarts, force, overwrite)


def _fit_validate_export(xml, out, path, logs, joints, restarts, force, overwrite) -> None:
    fit_config = FitConfig(max_restarts=restarts)
    with console.status("fitting (about 1-2 min per joint)..."):
        fits = pipeline.fit_run(xml, logs, joints, fit_config)
    pipeline.save_fits(fits, path / "fits.json")
    _print_fits(fits)
    results = pipeline.validate_run(xml, logs, fits, fit_config)
    pipeline.save_validation(results, path / "validation.json")
    _print_validation(results)
    target = out or xml.with_name(f"{xml.stem}_calibrated{xml.suffix}")
    result = export_calibrated(xml, target, fits, fit_config, force=force, overwrite=overwrite)
    for item in result.joints:
        console.print(
            f"{item.joint}: "
            + ("[green]exported[/green]" if item.exported else f"[yellow]skipped: {item.reason}[/yellow]")
        )
    console.print(f"calibrated model: [bold]{target}[/bold]   run data: {path}")


@app.command()
def demo(run_dir: RunDirOption = None) -> None:
    """The whole pipeline on a simulated 3-joint arm with known dynamics (no hardware, about 2-3 minutes)."""
    path = _run_dir(run_dir, prefix="demo-")
    xml = path / "mini_arm.xml"
    shutil.copy(MINI_ARM_XML, xml)
    bus, arm_calibration, clock = pipeline.open_bus(xml, True, DEFAULT_PORT, 1_000_000, None)
    joint = bus.joints[0]
    config = RecordConfig(joints=(joint,), amplitude_scale=0.8)
    console.print(
        f"simulated arm: {joint} has known dynamics (kp x0.7, kv x1.5, damping x1.4, ...) "
        f"and {DEMO_DELAY_S * 1e3:.0f} ms latency"
    )
    logs = pipeline.record(bus, arm_calibration, clock, xml, config, (2, 3), path, confirm=lambda report: True)
    _fit_validate_export(xml, None, path, logs, (joint,), 2, False, True)
    truth = MockArm(xml).nominal[joint].scaled(DEMO_TRUTH_SCALE)
    table = Table("parameter", "truth", "fitted", title="Demo ground truth (known only because this is a simulation)")
    fit = pipeline.load_fits(path / "fits.json")[0]
    for name, value in truth.to_dict().items():
        table.add_row(name, f"{value:.4g}", f"{getattr(fit.dynamics, name):.4g}")
    table.add_row("delay (ms)", f"{DEMO_DELAY_S * 1e3:.2f}", f"{fit.delay_s * 1e3:.2f}")
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
