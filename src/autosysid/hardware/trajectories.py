"""The three-stage excitation plan and the simulation pre-flight that must pass before any motion is commanded.

A `Segment` is a pure description: offsets from a centre pose on an exact `k / rate_hz` grid. Nothing here touches the
bus. The plan is deliberately conservative on first use (`amplitude_scale`), sized per joint later from real tracking.

  Stage 1  per joint, +-3 deg steps and slow ramps: checks the data path, sign/scale, latency, static behaviour.
  Stage 2  per joint, the ramp / chirp / saturating-steps recipe (docs/data-contract.md section 4), others held.
  Stage 3  all excited joints at once, multisine; one segment is HELD OUT for validation and never fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from autosysid.config import SafetyLimits
from autosysid.errors import ConfigError
from autosysid.excitation import PROTOCOL_PARTS, multisine, protocol_signal, step_train, triangle
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.safety import GoalLimiter
from autosysid.model import load_spec, scale_force_limits

STAGE1_AMPLITUDE_RAD = 0.05  # ~3 degrees: ~33 encoder ticks, clearly visible yet harmless
STAGE3_AMPLITUDE_RAD = 0.12  # per joint; with 5 joints moving this stays inside what the single-joint stage proved safe
STAGE3_DURATION_S = 30.0
STAGE3_FREQUENCIES_HZ = np.geomspace(0.15, 1.2, 6)  # below the ~1.5 Hz the servo tracks without saturating (section 4)
TRACKING_ERROR_BUDGET = 0.5  # predicted tracking error must stay below this fraction of the watchdog limit
SATURATION_WARNING = 0.05  # a chirp that saturates more than this is no longer an "unsaturated chirp"


@dataclass(frozen=True)
class Segment:
    name: str
    stage: str  # "stage1" | "stage2" | "stage3"
    role: str  # "fit" | "holdout"
    excited: tuple[str, ...]
    times_s: np.ndarray  # (N,) exact grid starting at 0
    offsets_rad: np.ndarray  # (N, J) in bus order; zero for held joints

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])


def _grid(duration_s: float, rate_hz: float) -> np.ndarray:
    return np.arange(int(round(duration_s * rate_hz)) + 1) / rate_hz


def build_plan(
    excited: tuple[str, ...],
    bus_joints: tuple[str, ...],
    rate_hz: float,
    amplitude_scale: float,
    limits: SafetyLimits,
    seed: int = 0,
    stages: tuple[int, ...] = (1, 2, 3),
) -> list[Segment]:
    """Segments in execution order. `amplitude_scale` scales amplitudes and step slew rate together."""
    unknown = [name for name in excited if name not in bus_joints]
    if unknown or not excited:
        raise ConfigError(f"excited joints {excited} must be a non-empty subset of the bus joints {bus_joints}")
    if not 0 < amplitude_scale <= 1:
        raise ConfigError("amplitude_scale must be in (0, 1]")
    max_rate = limits.max_goal_rate_rad_s
    rng = np.random.default_rng(seed)
    segments: list[Segment] = []

    def single(name: str, stage: str, joint: str, times: np.ndarray, signal: np.ndarray) -> Segment:
        offsets = np.zeros((len(times), len(bus_joints)))
        offsets[:, bus_joints.index(joint)] = signal
        return Segment(f"{stage}_{name}_{joint}", stage, "fit", (joint,), times, offsets)

    for joint in excited:  # bus order is proximal -> distal
        if 1 in stages:
            a = STAGE1_AMPLITUDE_RAD * amplitude_scale
            times = _grid(9.0, rate_hz)
            levels = np.array([a, -a, a, -a, 0.0])
            steps = step_train(times, levels, 1.5, max_rate * amplitude_scale)
            segments.append(single("step", "stage1", joint, times, steps))
            times = _grid(12.0, rate_hz)
            segments.append(single("ramp", "stage1", joint, times, triangle(times, a, 6.0)))
        if 2 in stages:
            for part, (label, duration) in PROTOCOL_PARTS.items():
                times = _grid(duration, rate_hz)
                signal = protocol_signal(part, times, amplitude_scale, min(max_rate, 2.0))
                segments.append(single(label, "stage2", joint, times, signal))
    if 3 in stages:
        for role in ("fit", "holdout"):
            times = _grid(STAGE3_DURATION_S, rate_hz)
            offsets = np.zeros((len(times), len(bus_joints)))
            for joint in excited:
                offsets[:, bus_joints.index(joint)] = multisine(
                    times, STAGE3_FREQUENCIES_HZ, STAGE3_AMPLITUDE_RAD * amplitude_scale, rng, taper_s=2.0
                )
            segments.append(Segment(f"stage3_multisine_{role}", "stage3", role, excited, times, offsets))
    return segments


def limited_goals(
    segment: Segment, center_rad: np.ndarray, calibration: ArmCalibration, limits: SafetyLimits
) -> tuple[np.ndarray, int, bool]:
    """What the safety limiter will actually send for `segment`: (goals (N, J) after tick quantization, #limited ticks,
    whether the request had to be clipped at a soft position limit)."""
    lower, upper = calibration.soft_limits_rad(limits.soft_limit_fraction)
    desired = center_rad + segment.offsets_rad
    clipped_at_bounds = bool(np.any(desired < lower - 1e-9) or np.any(desired > upper + 1e-9))
    dt = float(segment.times_s[1] - segment.times_s[0])
    limiter = GoalLimiter(limits, lower, upper, center_rad, dt)
    goals = np.array([limiter.step(row, dt) for row in desired])
    quantized = np.array([calibration.ticks_to_rad(calibration.rad_to_ticks(row, None)) for row in goals])
    return quantized, limiter.n_limited, clipped_at_bounds


@dataclass
class SegmentReport:
    name: str
    max_command_rate_rad_s: float = 0.0
    limited_fraction: float = 0.0
    max_tracking_error_rad: float = 0.0
    saturated_fraction: dict[str, float] = field(default_factory=dict)
    new_contacts: int = 0
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreflightReport:
    segments: list[SegmentReport]

    @property
    def ok(self) -> bool:
        return all(not report.violations for report in self.segments)

    def summary(self) -> str:
        lines = []
        for report in self.segments:
            status = "FAIL" if report.violations else ("warn" if report.warnings else "ok")
            saturation = max(report.saturated_fraction.values(), default=0.0)
            lines.append(
                f"{status:<5}{report.name:<34} rate {report.max_command_rate_rad_s:4.2f} rad/s  "
                f"track {report.max_tracking_error_rad * 1e3:5.1f} mrad  sat {saturation:4.0%}"
                + "".join(f"\n        ! {message}" for message in report.violations + report.warnings)
            )
        return "\n".join(lines)


def preflight(
    plan: list[Segment],
    xml_path: str | Path,
    bus_joints: tuple[str, ...],
    center_rad: np.ndarray,
    calibration: ArmCalibration,
    limits: SafetyLimits,
) -> PreflightReport:
    """Simulate every segment on the *nominal* model from `center_rad`, exactly as the limiter will send it.

    Violations (block the run): request beyond a soft limit, hitting the model's joint range, a contact that does not
    exist at the start pose (self-collision / floor), predicted tracking error above half the watchdog limit.
    Warnings: limiter engaged often, chirp saturating the actuator (it would no longer be a linear-regime signal).
    """
    spec = load_spec(xml_path, disable_contacts=False)
    # the arm runs with a reduced Torque_Limit (SafetySession), so predict saturation with the same clamp
    scale_force_limits(spec, {name: limits.torque_limit_fraction for name in bus_joints})
    model = spec.compile()
    missing = [name for name in bus_joints if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0]
    if missing:
        raise ConfigError(f"model has no actuator for bus joints {missing}")
    qadr = np.array([model.joint(name).qposadr[0] for name in bus_joints])
    act = np.array([model.actuator(name).id for name in bus_joints])
    force_limit = model.actuator_forcerange[act, 1]
    data = mujoco.MjData(model)
    data.qpos[qadr] = center_rad
    mujoco.mj_forward(model, data)
    baseline_contacts = {_pair(data.contact[i]) for i in range(data.ncon)}
    initial_state = (data.qpos.copy(), data.qvel.copy())
    reports = []
    for segment in plan:
        report = SegmentReport(segment.name)
        goals, n_limited, clipped = limited_goals(segment, center_rad, calibration, limits)
        dt = float(segment.times_s[1] - segment.times_s[0])
        report.max_command_rate_rad_s = float(np.max(np.abs(np.diff(goals, axis=0))) / dt)
        report.limited_fraction = n_limited / len(goals)
        if clipped:
            report.violations.append(
                "request exceeds a soft position limit from this start pose (centre the arm or reduce amplitude)"
            )
        if report.limited_fraction > 0.02:
            report.warnings.append(f"rate/acceleration limiter engaged on {report.limited_fraction:.0%} of ticks")
        data.qpos[:], data.qvel[:] = initial_state
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        columns = [bus_joints.index(name) for name in segment.excited]
        tracking, saturated = np.zeros(len(columns)), np.zeros(len(columns))
        steps_per_tick = max(int(round(dt / model.opt.timestep)), 1)
        contacts: set = set()
        out_of_range = False
        for goal in goals:
            data.ctrl[act] = goal
            for _ in range(steps_per_tick):
                mujoco.mj_step(model, data)
                contacts.update(_pair(data.contact[i]) for i in range(data.ncon))
            error = np.abs(goal[columns] - data.qpos[qadr[columns]])
            tracking = np.maximum(tracking, error)
            saturated += np.abs(data.actuator_force[act[columns]]) >= 0.999 * force_limit[columns]
            low, high = model.jnt_range[[model.joint(bus_joints[c]).id for c in columns]].T
            out_of_range |= bool(np.any(data.qpos[qadr[columns]] < low) or np.any(data.qpos[qadr[columns]] > high))
        report.max_tracking_error_rad = float(np.max(tracking))
        report.saturated_fraction = {name: float(saturated[i] / len(goals)) for i, name in enumerate(segment.excited)}
        report.new_contacts = len(contacts - baseline_contacts)
        if report.new_contacts:
            report.violations.append(
                f"{report.new_contacts} new contact pair(s) in simulation (self-collision or floor)"
            )
        if out_of_range:
            report.violations.append("simulation reaches the model's joint range limit")
        if report.max_tracking_error_rad > TRACKING_ERROR_BUDGET * limits.max_tracking_error_rad:
            report.violations.append(
                f"predicted tracking error {report.max_tracking_error_rad:.2f} rad would trip the watchdog "
                f"({limits.max_tracking_error_rad} rad)"
            )
        if "chirp" in segment.name and max(report.saturated_fraction.values()) > SATURATION_WARNING:
            report.warnings.append("chirp saturates the actuator; reduce amplitude_scale")
        reports.append(report)
    return PreflightReport(reports)


def _pair(contact) -> tuple[int, int]:
    return (int(min(contact.geom1, contact.geom2)), int(max(contact.geom1, contact.geom2)))


if __name__ == "__main__":
    from autosysid.mock import MINI_ARM_XML

    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex")
    calibration = ArmCalibration.symmetric(joints)
    safety = SafetyLimits()
    plan = build_plan(("shoulder_pan",), joints, 100.0, 0.5, safety, stages=(2,))
    report = preflight(plan, MINI_ARM_XML, joints, np.zeros(3), calibration, safety)
    print(report.summary())
    assert report.ok, "a half-amplitude single-joint plan should pass pre-flight"
    print("trajectory sanity checks passed")
