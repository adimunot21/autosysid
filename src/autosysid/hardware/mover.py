"""Commanded single moves of the arm to a target pose, through the same safety layer as every recording.

Why this exists: posing the arm by hand needs torque OFF, and a released gravity-loaded arm falls (shoulder_lift sagged
47 deg). A commanded move keeps torque ON throughout, hits the pose exactly, and is rate- and limit-gated.

Guarantees (each is a test in tests/test_mover.py):
  * targets are validated against the SOFT limits and refused, never clipped silently;
  * speed is capped at `rate_rad_s` (default 0.3 rad/s, as in `safe_stop`'s return), never above the safety cap;
  * goals pass through `GoalLimiter`, and every tick through `SafetyMonitor` (tracking error, temperature, voltage,
    loop jitter, comm loss), exactly as in `Recorder`;
  * torque is never released: on success the arm holds the new pose, on a trip it returns to where it started and holds.

What it cannot know: where the table is. The caller (a human) must confirm the path is clear.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from autosysid.config import SafetyLimits
from autosysid.errors import ConfigError, HardwareError, SafetyViolation
from autosysid.hardware.bus import ServoBus
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import Clock
from autosysid.hardware.safety import GoalLimiter, SafetyMonitor, SafetySession
from autosysid.logging_utils import get_logger

log = get_logger("mover")

CONFIG = {
    "rate_hz": 100.0,  # same loop rate as the recorder: the watchdog thresholds were tuned for it
    "default_rate_rad_s": 0.3,  # ~17 deg/s: slow enough to watch and to stop by hand, same as safe_stop's return
    "settle_s": 1.0,  # hold at the goal so the reported end pose is the resting one, not a moving one
    "timeout_factor": 3.0,  # abort a move that takes this many times longer than distance / rate (a stall)
}


@dataclass(frozen=True)
class MoveResult:
    start_rad: np.ndarray  # present pose before the move, in bus joint order
    goal_rad: np.ndarray  # the commanded final pose (targets merged over the start pose)
    end_rad: np.ndarray  # present pose after the settle; differs from goal by the servo's deadband and gravity sag
    duration_s: float


def validate_targets(
    targets: Mapping[str, float], joints: tuple[str, ...], lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    """Check names, finiteness and soft limits; returns a plain dict. Raises `ConfigError` with the offending joint."""
    if not targets:
        raise ConfigError("no target joints given")
    checked: dict[str, float] = {}
    for name, value in targets.items():
        if name not in joints:
            raise ConfigError(f"unknown joint '{name}'; available: {list(joints)}")
        if not math.isfinite(value):
            raise ConfigError(f"target for {name} must be finite, got {value}")
        index = joints.index(name)
        if not lower[index] <= value <= upper[index]:
            raise ConfigError(
                f"target {value:+.3f} rad for {name} is outside its soft limit "
                f"[{lower[index]:+.3f}, {upper[index]:+.3f}] rad"
            )
        checked[name] = float(value)
    return checked


def move_to(
    bus: ServoBus,
    calibration: ArmCalibration,
    limits: SafetyLimits,
    clock: Clock,
    targets_rad: Mapping[str, float],
    rate_rad_s: float = CONFIG["default_rate_rad_s"],
    progress: Callable[[np.ndarray, np.ndarray], None] | None = None,
) -> MoveResult:
    """Move the named joints to `targets_rad` (radians, MJCF frame), holding every other joint where it is."""
    if not (math.isfinite(rate_rad_s) and 0 < rate_rad_s <= limits.max_goal_rate_rad_s):
        raise ConfigError(f"rate_rad_s must be in (0, {limits.max_goal_rate_rad_s}], got {rate_rad_s}")
    lower, upper = calibration.soft_limits_rad(limits.soft_limit_fraction)
    targets = validate_targets(targets_rad, tuple(bus.joints), lower, upper)  # before anything is energised

    hold_limits = replace(limits, release_torque_at_end=False, max_goal_rate_rad_s=rate_rad_s)
    rate_hz = CONFIG["rate_hz"]
    dt = 1.0 / rate_hz
    started = clock.now()
    with SafetySession(bus, calibration, hold_limits, rate_hz, clock) as session:
        start = session.start_pose_rad.copy()
        goal = start.copy()
        for name, value in targets.items():
            goal[list(bus.joints).index(name)] = value
        distance = float(np.max(np.abs(goal - start)))
        log.info("moving %s: %s -> %s rad", list(targets), np.round(start, 3).tolist(), np.round(goal, 3).tolist())

        limiter = GoalLimiter(hold_limits, lower, upper, start, dt)
        monitor = SafetyMonitor(hold_limits, bus.joints, calibration)
        n_ticks = int(CONFIG["timeout_factor"] * distance / rate_hz_to_step(rate_rad_s, dt)) + int(
            CONFIG["settle_s"] * rate_hz
        )
        deadline = clock.now() + dt
        settled_ticks = 0
        for _ in range(max(n_ticks, int(CONFIG["settle_s"] * rate_hz))):
            clock.sleep_until(deadline)
            if session.stop_requested:
                raise SafetyViolation("interrupted by operator")
            last_goal = limiter.goal.copy()
            try:
                snapshot = bus.read_state()
            except HardwareError:
                monitor.note_comm_failure(clock.now())  # raises once communication has been lost for too long
                deadline += dt
                continue
            measured = calibration.ticks_to_rad(snapshot.position_ticks)
            monitor.check_tick(snapshot.t_start_s - deadline)  # watchdog trips are SafetyViolation (a HardwareError):
            monitor.check_state(snapshot, measured, last_goal)  # they must stay OUTSIDE the try above
            ticks = calibration.rad_to_ticks(limiter.step(goal, dt))
            try:
                bus.write_goal_ticks(ticks)
            except HardwareError:
                monitor.note_comm_failure(clock.now())
                deadline += dt
                continue
            if progress is not None:
                progress(measured, limiter.goal)
            at_goal = float(np.max(np.abs(limiter.goal - goal))) < 1e-4
            settled_ticks = settled_ticks + 1 if at_goal else 0
            if settled_ticks >= CONFIG["settle_s"] * rate_hz:
                break
            deadline += dt
        else:
            raise SafetyViolation(f"move did not complete within the expected time (goal {np.round(goal, 3)} rad)")

        end = calibration.ticks_to_rad(bus.read_state().position_ticks)
        session.start_pose_rad = goal  # on exit the session parks at its start pose: that is now the new pose
    return MoveResult(start_rad=start, goal_rad=goal, end_rad=end, duration_s=clock.now() - started)


def rate_hz_to_step(rate_rad_s: float, dt: float) -> float:
    """Radians per control tick at the capped rate."""
    return rate_rad_s * dt


if __name__ == "__main__":
    from autosysid.hardware.bus import MockBus
    from autosysid.hardware.clock import SimClock

    names = ("shoulder_pan", "shoulder_lift", "elbow_flex")
    sim_clock = SimClock()
    calibration_demo = ArmCalibration.symmetric(names)
    mock = MockBus(calibration_demo, sim_clock)
    result = move_to(mock, calibration_demo, SafetyLimits(), sim_clock, {"shoulder_lift": 0.4})
    print(
        f"start {np.round(result.start_rad, 3)}  goal {np.round(result.goal_rad, 3)}  end {np.round(result.end_rad, 3)}"
    )
    print(f"took {result.duration_s:.1f} s (virtual), torque still on: {mock.torque_enabled}")
    assert abs(result.end_rad[1] - 0.4) < 0.02 and mock.torque_enabled
    print("ok")
