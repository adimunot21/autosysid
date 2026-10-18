"""Safety layer: command limiter, state watchdogs, and a session that always leaves the arm held, parked and restored.

Design rules (why each exists):
  * Every goal passes through `GoalLimiter` (soft position limits, velocity cap, acceleration cap): no trajectory
    generator can command something the arm was not cleared for, and a bug upstream cannot snap a plastic bracket.
  * `SafetyMonitor` trips on facts the servo reports (temperature, voltage), on the *outcome* (tracking error: stall,
    collision or runaway) and on the loop itself (stalled loop, lost communication). Load/current are deliberately NOT
    used until their units are verified on hardware (docs/data-contract.md); tracking error is the proxy.
  * Cutting torque drops the arm onto the table, so a trip first holds, then returns slowly to the start pose, and only
    then releases torque. If communication is lost we cannot do any of that; we leave torque ON (the servos hold the
    last
    goal) and say so loudly, because holding is safer than dropping.
  * Registers we change (`torque_limit`, `acceleration`) are restored in `finally`, whatever happened.
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass, field

import numpy as np

from autosysid.config import SafetyLimits
from autosysid.errors import HardwareError, SafetyViolation
from autosysid.hardware.bus import ServoBus, ServoSnapshot
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import Clock, RealClock
from autosysid.logging_utils import get_logger

log = get_logger("safety")
_BRAKING_MARGIN = (
    0.5  # plan the stop with half the acceleration cap: discrete ticks cannot follow the ideal sqrt profile
)
_MAX_DT_FACTOR = 3.0  # after a stalled tick, the limiter integrates at most this many nominal periods


class GoalLimiter:
    """Turns desired goals into admissible goals: inside [lower, upper], |v| <= max rate, |a| <= max acceleration."""

    def __init__(
        self, limits: SafetyLimits, lower: np.ndarray, upper: np.ndarray, initial: np.ndarray, dt: float
    ) -> None:
        self.limits, self.lower, self.upper, self.nominal_dt = limits, lower, upper, dt
        self.goal = np.clip(np.asarray(initial, dtype=float), lower, upper)
        self.velocity = np.zeros_like(self.goal)
        self._previous_desired = self.goal.copy()
        self.tolerance_rad = 0.002  # output deviating from the request by more than this (~1 tick) counts as limited
        self.n_limited = 0  # ticks on which the output differed from the request: a plan that needs this is a bad plan

    def step(self, desired: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = min(dt or self.nominal_dt, _MAX_DT_FACTOR * self.nominal_dt)
        desired = np.clip(desired, self.lower, self.upper)
        # Feed-forward the target's own velocity and close the remaining lag to the PREVIOUS target (the feed-forward
        # already covers this tick's motion), capping the closing speed at the braking speed sqrt(2*a*|lag|). Without
        # the
        # cap a lagging goal chases its target at full speed and overshoots; without the feed-forward the cap would
        # throttle a smoothly moving target.
        target_velocity = (desired - self._previous_desired) / dt
        lag = self._previous_desired - self.goal
        closing_speed = np.sign(lag) * np.minimum(
            np.abs(lag) / dt, np.sqrt(2.0 * _BRAKING_MARGIN * self.limits.max_goal_accel_rad_s2 * np.abs(lag))
        )
        self._previous_desired = desired
        wanted_velocity = np.clip(
            target_velocity + closing_speed, -self.limits.max_goal_rate_rad_s, self.limits.max_goal_rate_rad_s
        )
        max_dv = self.limits.max_goal_accel_rad_s2 * dt
        velocity = np.clip(wanted_velocity, self.velocity - max_dv, self.velocity + max_dv)
        new_goal = np.clip(self.goal + velocity * dt, self.lower, self.upper)
        self.velocity = (new_goal - self.goal) / dt
        self.goal = new_goal
        if np.max(np.abs(new_goal - desired)) > self.tolerance_rad:
            self.n_limited += 1
        return new_goal.copy()


class SafetyMonitor:
    """Raises `SafetyViolation` with a specific reason. Call `check_state` after every successful read."""

    def __init__(self, limits: SafetyLimits, joints: tuple[str, ...], calibration: ArmCalibration) -> None:
        self.limits, self.joints = limits, joints
        self.hard_lower, self.hard_upper = calibration.hard_limits_rad()
        self._first_failure_s: float | None = None

    def check_state(self, snapshot: ServoSnapshot, measured_rad: np.ndarray, last_goal_rad: np.ndarray | None) -> None:
        self._first_failure_s = None  # a good read ends any comm-failure streak
        hot = np.flatnonzero(snapshot.temperature_c > self.limits.max_temperature_c)
        if hot.size:
            raise SafetyViolation(
                f"over-temperature: {self.joints[hot[0]]} at {snapshot.temperature_c[hot[0]]:.0f} C "
                f"(limit {self.limits.max_temperature_c} C)"
            )
        if np.min(snapshot.voltage_v) < self.limits.min_voltage_v:
            raise SafetyViolation(
                f"supply voltage {np.min(snapshot.voltage_v):.1f} V below {self.limits.min_voltage_v} V"
            )
        outside = np.flatnonzero((measured_rad < self.hard_lower - 0.05) | (measured_rad > self.hard_upper + 0.05))
        if outside.size:
            raise SafetyViolation(
                f"{self.joints[outside[0]]} at {measured_rad[outside[0]]:.3f} rad is outside its calibrated range"
            )
        if last_goal_rad is not None:
            error = np.abs(last_goal_rad - measured_rad)
            worst = int(np.argmax(error))
            if error[worst] > self.limits.max_tracking_error_rad:
                raise SafetyViolation(
                    f"tracking error {error[worst]:.3f} rad on {self.joints[worst]} exceeds "
                    f"{self.limits.max_tracking_error_rad} rad (stall, collision or runaway)"
                )

    def check_tick(self, lateness_s: float) -> None:
        if lateness_s > self.limits.max_loop_jitter_s:
            raise SafetyViolation(
                f"control loop late by {lateness_s * 1e3:.1f} ms (limit {self.limits.max_loop_jitter_s * 1e3:.0f} ms)"
            )

    def note_comm_failure(self, now_s: float) -> None:
        if self._first_failure_s is None:
            self._first_failure_s = now_s
        if now_s - self._first_failure_s >= self.limits.comm_timeout_s:
            raise SafetyViolation(f"no valid servo response for {now_s - self._first_failure_s:.2f} s")


def safe_stop(
    bus: ServoBus,
    calibration: ArmCalibration,
    limits: SafetyLimits,
    clock: Clock,
    rest_rad: np.ndarray,
    rate_hz: float,
    release: bool = True,
) -> bool:
    """Hold, return slowly to `rest_rad`, then (if `release`) torque off. True if the arm was parked and released.

    Best effort and never raises: it runs while handling another failure. On lost communication the arm is left holding.
    """
    dt = 1.0 / rate_hz
    try:
        snapshot = bus.read_state()
    except HardwareError as error:
        log.error(
            "SAFE STOP: cannot read the arm (%s). Torque left ON; the servos hold their last goal. Power off manually.",
            error,
        )
        return False
    present = calibration.ticks_to_rad(snapshot.position_ticks)
    lower, upper = calibration.hard_limits_rad()
    goal = present.copy()
    try:
        bus.write_goal_ticks(calibration.rad_to_ticks(goal))  # hold exactly where we are first
        rest = np.clip(rest_rad, lower, upper)
        step = limits.safe_stop_rate_rad_s * dt
        deadline = clock.now() + 30.0
        while np.max(np.abs(rest - goal)) > 1e-4 and clock.now() < deadline:
            clock.sleep_until(clock.now() + dt)
            goal = goal + np.clip(rest - goal, -step, step)
            bus.write_goal_ticks(calibration.rad_to_ticks(goal))
        clock.sleep_until(clock.now() + 0.3)  # let the arm settle on the last goal before letting go
        if release:
            bus.set_torque(False)
        return release
    except HardwareError as error:
        log.error("SAFE STOP interrupted by a bus error (%s). Torque state unknown; check the arm.", error)
        return False


@dataclass
class SafetySession:
    """Context manager around everything that moves the arm. Entering reads the arm and enables torque *at the present
    pose* (never at a stale goal, which would jerk it); leaving always parks, releases and restores registers."""

    bus: ServoBus
    calibration: ArmCalibration
    limits: SafetyLimits
    rate_hz: float
    clock: Clock = field(default_factory=RealClock)
    start_pose_rad: np.ndarray = field(init=False)
    stop_requested: bool = field(default=False, init=False)
    torque_scale: dict[str, float] = field(
        default_factory=dict, init=False
    )  # Torque_Limit as a fraction of what it was
    _saved_settings: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _previous_handler: object = field(default=None, init=False)
    _engaged: bool = field(default=False, init=False)

    def __enter__(self) -> SafetySession:
        snapshot = self.bus.read_state()
        SafetyMonitor(self.limits, self.bus.joints, self.calibration).check_state(
            snapshot, self.calibration.ticks_to_rad(snapshot.position_ticks), None
        )
        self.start_pose_rad = self.calibration.ticks_to_rad(snapshot.position_ticks)
        lower, upper = self.calibration.soft_limits_rad(self.limits.soft_limit_fraction)
        if np.any(self.start_pose_rad < lower) or np.any(self.start_pose_rad > upper):
            raise SafetyViolation(
                "start pose is outside the soft limits; move the arm to the middle of its range first "
                f"(pose {np.round(self.start_pose_rad, 2).tolist()} rad)"
            )
        if threading.current_thread() is threading.main_thread():
            self._previous_handler = signal.signal(signal.SIGINT, self._on_interrupt)
        try:
            for name in ("torque_limit", "acceleration"):
                self._saved_settings[name] = self.bus.read_setting(name)
            reduced = np.maximum(
                (self._saved_settings["torque_limit"] * self.limits.torque_limit_fraction).astype(int), 1
            )
            self.bus.write_setting("torque_limit", reduced)
            self.torque_scale = dict(
                zip(self.bus.joints, (reduced / self._saved_settings["torque_limit"]).tolist(), strict=True)
            )
            self.bus.write_goal_ticks(snapshot.position_ticks)
            self.bus.set_torque(True)
            self._engaged = True
        except BaseException:
            self._cleanup(engaged=self._engaged)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            log.error("stopping because of %s: %s", exc_type.__name__, exc)
        self._cleanup(engaged=self._engaged)
        return False  # never swallow the original exception

    def _on_interrupt(self, signum, frame) -> None:
        self.stop_requested = True

    def _cleanup(self, engaged: bool) -> None:
        try:
            if engaged:
                safe_stop(self.bus, self.calibration, self.limits, self.clock, self.start_pose_rad, self.rate_hz)
            for name, values in self._saved_settings.items():
                try:
                    self.bus.write_setting(name, values)
                except HardwareError as error:
                    log.error("could not restore %s (%s); power-cycle the arm to reset it", name, error)
        finally:
            if self._previous_handler is not None:
                signal.signal(signal.SIGINT, self._previous_handler)  # type: ignore[arg-type]
                self._previous_handler = None


if __name__ == "__main__":
    limits = SafetyLimits()
    limiter = GoalLimiter(limits, np.array([-1.0]), np.array([1.0]), np.zeros(1), dt=0.01)
    outputs = np.array([limiter.step(np.array([5.0]))[0] for _ in range(300)])
    assert outputs.max() <= 1.0 + 1e-12, "soft limit violated"
    assert np.max(np.abs(np.diff(outputs))) <= limits.max_goal_rate_rad_s * 0.01 + 1e-12, "rate limit violated"
    assert limiter.n_limited > 0
    print(f"limiter sanity checks passed (reached {outputs[-1]:.2f} rad, {limiter.n_limited} limited ticks)")
