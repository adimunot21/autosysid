"""Orchestration shared by the CLI commands: build a bus, run a recording, select logs, fit, validate, persist.

Run directory layout (everything needed to reproduce or re-fit without the arm):
    <run>/raw/<segment>.npz   one RawLog per segment (targets, measurements, per-tick diagnostics)
    <run>/record.json         the RecordConfig and plan summary used
    <run>/fits.json           one JointFit per fitted joint
    <run>/validation.json     held-out prediction errors
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from autosysid.config import RecordConfig
from autosysid.data import RawLog
from autosysid.errors import ConfigError, DataError
from autosysid.hardware.bus import MockBus, ServoBus
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import Clock, RealClock, SimClock
from autosysid.hardware.feetech import FeetechBus
from autosysid.hardware.recorder import Progress, Recorder
from autosysid.hardware.trajectories import PreflightReport, build_plan
from autosysid.logging_utils import get_logger
from autosysid.mock import DEMO_DELAY_S, DEMO_TRUTH_SCALE, MockArm
from autosysid.optimizer import JointFit, Validation, fit_joint, validate_joint

log = get_logger("pipeline")
DEFAULT_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"
FIT_STAGE = "stage2"  # per-joint recipe; stage 1 is too small to constrain dynamics, stage 3 is coupled (held out)


def parse_joints(text: str | None, available: Sequence[str]) -> tuple[str, ...]:
    if not text:
        return tuple(available)
    joints = tuple(name.strip() for name in text.split(",") if name.strip())
    unknown = [name for name in joints if name not in available]
    if unknown:
        raise ConfigError(f"unknown joints {unknown}; available: {list(available)}")
    return joints


def parse_stages(text: str) -> tuple[int, ...]:
    try:
        stages = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise ConfigError(f"--stages must look like 1,2,3 (got '{text}')") from error
    if not stages or any(stage not in (1, 2, 3) for stage in stages):
        raise ConfigError(f"--stages must be a subset of 1,2,3 (got '{text}')")
    return stages


def open_bus(
    xml_path: Path, mock: bool, port: str, baudrate: int, calibration_path: Path | None
) -> tuple[ServoBus, ArmCalibration, Clock]:
    """A mock bus (no hardware; joints from the model, demo truth) or a real Feetech bus (opened, read-only)."""
    if mock:
        clock = SimClock()
        arm = MockArm(xml_path)
        calibration = ArmCalibration.symmetric(arm.joint_names)
        truth = {name: arm.nominal[name].scaled(DEMO_TRUTH_SCALE) for name in arm.joint_names[:1]}
        delays = {arm.joint_names[0]: DEMO_DELAY_S}
        return MockBus(calibration, clock, xml_path, truth=truth, delays_s=delays), calibration, clock
    calibration = ArmCalibration.from_lerobot_json(calibration_path or DEFAULT_CALIBRATION)
    clock = RealClock()
    return FeetechBus(port, baudrate, calibration, clock=clock).open(), calibration, clock


def record(
    bus: ServoBus,
    calibration: ArmCalibration,
    clock: Clock,
    xml_path: Path,
    config: RecordConfig,
    stages: tuple[int, ...],
    run_dir: Path,
    progress: Progress | None = None,
    confirm: Callable[[PreflightReport], bool] | None = None,
) -> list[RawLog]:
    plan = build_plan(
        config.joints, bus.joints, config.control_rate_hz, config.amplitude_scale, config.safety, config.seed, stages
    )
    logs = Recorder(bus, calibration, config, clock).run(plan, xml_path=xml_path, progress=progress, confirm=confirm)
    for recorded in logs:
        recorded.save(run_dir / "raw" / f"{recorded.stage}.npz")
    (run_dir / "record.json").write_text(
        json.dumps({"config": asdict(config), "stages": list(stages), "segments": [s.name for s in plan]}, indent=2)
    )
    return logs


def load_logs(run_dir: Path) -> list[RawLog]:
    files = sorted((run_dir / "raw").glob("*.npz"))
    if not files:
        raise DataError(f"no recorded logs in {run_dir / 'raw'}")
    return [RawLog.load(file) for file in files]


def excited_joints(logs: Sequence[RawLog]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for recorded in logs:
        for joint in recorded.meta.get("excited", []):
            seen[joint] = None
    return tuple(seen)


def fit_logs_for(logs: Sequence[RawLog], joint: str) -> list[RawLog]:
    return [
        recorded
        for recorded in logs
        if recorded.meta.get("stage") == FIT_STAGE and recorded.meta.get("excited") == [joint]
    ]


def holdout_logs_for(logs: Sequence[RawLog], joint: str) -> list[RawLog]:
    return [
        recorded
        for recorded in logs
        if recorded.meta.get("role") == "holdout" and joint in recorded.meta.get("excited", [])
    ]


def fit_run(xml_path: Path, logs: Sequence[RawLog], joints: Sequence[str], fit_config) -> list[JointFit]:
    fits = []
    for joint in joints:
        selected = fit_logs_for(logs, joint)
        if not selected:
            raise DataError(f"no {FIT_STAGE} logs for '{joint}' in this run (record with stage 2)")
        log.info("fitting %s on %d log(s)", joint, len(selected))
        fits.append(fit_joint(xml_path, selected, joint, fit_config))
    return fits


def validate_run(xml_path: Path, logs: Sequence[RawLog], fits: Sequence[JointFit], fit_config) -> list[Validation]:
    results = []
    for fit in fits:
        held_out = holdout_logs_for(logs, fit.joint)
        if held_out:
            results.append(validate_joint(xml_path, fit, held_out, fit_config))
        else:
            log.warning("no held-out log for %s (record with stage 3); skipping validation", fit.joint)
    return results


def save_fits(fits: Sequence[JointFit], path: Path) -> None:
    path.write_text(json.dumps({fit.joint: fit.to_dict() for fit in fits}, indent=2))


def load_fits(path: Path) -> list[JointFit]:
    try:
        return [JointFit.from_dict(raw) for raw in json.loads(path.read_text()).values()]
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise DataError(f"cannot read fits {path}: {error}") from error


def save_validation(results: Sequence[Validation], path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                r.joint: {
                    "rms_nominal_rad": r.rms_nominal_rad,
                    "rms_fitted_rad": r.rms_fitted_rad,
                    "improvement": r.improvement,
                    "n_samples": r.n_samples,
                }
                for r in results
            },
            indent=2,
        )
    )
