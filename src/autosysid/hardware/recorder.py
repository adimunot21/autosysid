"""Fixed-rate record loop: read state -> check safety -> limit goal -> write goal -> log, one row per tick.

Row semantics match `data.RawLog` and the simulator: row i holds the state read at (about) `times[i]` and the
goal written right after it, which then acts from `times[i]` on. The read-to-write gap (~1.5 ms) is therefore
absorbed into the fitted
delay. The logged target is what was actually sent (limited, then quantized to servo ticks), never the plan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autosysid.config import RecordConfig
from autosysid.data import RawLog
from autosysid.errors import Aborted, HardwareError, SafetyViolation
from autosysid.hardware.bus import ServoBus
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import Clock, RealClock
from autosysid.hardware.safety import GoalLimiter, SafetyMonitor, SafetySession
from autosysid.hardware.trajectories import PreflightReport, Segment, preflight
from autosysid.logging_utils import get_logger

log = get_logger("recorder")
SETTLE_S = 1.0  # hold at the centre pose between segments so each one starts at rest
START_DELAY_S = 0.05  # first tick this long after the segment begins (clock + first-read warm-up)

Progress = Callable[[int, int, str], None]


class Recorder:
    def __init__(
        self, bus: ServoBus, calibration: ArmCalibration, config: RecordConfig, clock: Clock | None = None
    ) -> None:
        self.bus, self.calibration, self.config = bus, calibration, config
        self.clock: Clock = clock or RealClock()
        self.rate_hz = config.control_rate_hz

    def read_pose_rad(self) -> np.ndarray:
        """Present pose without engaging torque or writing anything (used to centre the plan and run pre-flight)."""
        return self.calibration.ticks_to_rad(self.bus.read_state().position_ticks)

    def run(
        self,
        plan: list[Segment],
        xml_path: str | Path | None = None,
        progress: Progress | None = None,
        confirm: Callable[[PreflightReport], bool] | None = None,
    ) -> list[RawLog]:
        """Execute `plan` and return one log per segment. With `xml_path`, runs the simulation pre-flight first and
        refuses to move the arm if it fails."""
        limits = self.config.safety
        if xml_path is not None:
            report = preflight(plan, xml_path, self.bus.joints, self.read_pose_rad(), self.calibration, limits)
            log.info("pre-flight:\n%s", report.summary())
            if not report.ok:
                raise SafetyViolation("pre-flight failed; nothing was moved:\n" + report.summary())
            if confirm is not None and not confirm(report):
                raise Aborted("declined by operator; nothing was moved")
        logs: list[RawLog] = []
        with SafetySession(self.bus, self.calibration, limits, self.rate_hz, self.clock) as session:
            lower, upper = self.calibration.soft_limits_rad(limits.soft_limit_fraction)
            limiter = GoalLimiter(limits, lower, upper, session.start_pose_rad, 1.0 / self.rate_hz)
            monitor = SafetyMonitor(limits, self.bus.joints, self.calibration)
            center = session.start_pose_rad
            for index, segment in enumerate(plan):
                if progress:
                    progress(index, len(plan), segment.name)
                self._settle(session, limiter, monitor, center)
                logs.append(self._run_segment(session, limiter, monitor, segment, center))
            self._settle(session, limiter, monitor, center)
        return logs

    def _tick(
        self,
        session: SafetySession,
        limiter: GoalLimiter,
        monitor: SafetyMonitor,
        desired: np.ndarray,
        deadline_s: float,
    ):
        """One control tick. Returns (snapshot, measured_rad, sent_goal_rad) or None if the bus failed this tick."""
        self.clock.sleep_until(deadline_s)
        if session.stop_requested:
            raise SafetyViolation("interrupted by operator")
        last_goal = limiter.goal.copy()
        try:
            snapshot = self.bus.read_state()
        except HardwareError:
            monitor.note_comm_failure(self.clock.now())
            return None
        measured = self.calibration.ticks_to_rad(snapshot.position_ticks)
        monitor.check_tick(snapshot.t_start_s - deadline_s)
        monitor.check_state(snapshot, measured, last_goal)
        ticks = self.calibration.rad_to_ticks(limiter.step(desired, 1.0 / self.rate_hz))
        try:
            self.bus.write_goal_ticks(ticks)
        except HardwareError:
            monitor.note_comm_failure(self.clock.now())
            return None
        return snapshot, measured, self.calibration.ticks_to_rad(ticks)

    def _settle(self, session: SafetySession, limiter: GoalLimiter, monitor: SafetyMonitor, center: np.ndarray) -> None:
        deadline = self.clock.now() + START_DELAY_S
        n_ticks = int(SETTLE_S * self.rate_hz) + int(np.max(np.abs(limiter.goal - center)) / 0.2 * self.rate_hz)
        for _ in range(n_ticks):
            self._tick(session, limiter, monitor, center, deadline)
            deadline += 1.0 / self.rate_hz

    def _run_segment(
        self, session: SafetySession, limiter: GoalLimiter, monitor: SafetyMonitor, segment: Segment, center: np.ndarray
    ) -> RawLog:
        dt = 1.0 / self.rate_hz
        rows: list[tuple] = []
        failures = 0
        deadline = self.clock.now() + START_DELAY_S
        limited_before = limiter.n_limited
        for offsets in segment.offsets_rad:
            result = self._tick(session, limiter, monitor, center + offsets, deadline)
            if result is None:
                failures += 1
            else:
                snapshot, measured, sent = result
                rows.append((snapshot, measured, sent, snapshot.t_start_s - deadline))
            deadline += dt
        if len(rows) < 0.9 * len(segment.offsets_rad):
            raise SafetyViolation(f"{segment.name}: only {len(rows)}/{len(segment.offsets_rad)} ticks succeeded")
        snapshots = [row[0] for row in rows]
        return RawLog(
            times=np.array([s.t_mid_s for s in snapshots]),
            joints=tuple(self.bus.joints),
            target_rad=np.array([row[2] for row in rows]),
            measured_rad=np.array([row[1] for row in rows]),
            stage=segment.name,
            meta={
                "segment": segment.name,
                "stage": segment.stage,
                "role": segment.role,
                "excited": list(segment.excited),
                "center_rad": center.tolist(),
                # the servo saturates earlier with a reduced Torque_Limit; the fit must clamp the model the same way
                "torque_scale": session.torque_scale,
                "rate_hz": self.rate_hz,
                "limited_ticks": limiter.n_limited - limited_before,
                "comm_failures": failures,
                "record_config": asdict(self.config),
            },
            extras={
                "velocity_raw": np.array([s.velocity_raw for s in snapshots]),
                "load_raw": np.array([s.load_raw for s in snapshots]),
                "temperature_c": np.array([s.temperature_c for s in snapshots]),
                "voltage_v": np.array([s.voltage_v for s in snapshots]),
                "read_duration_s": np.array([s.read_duration_s for s in snapshots]),
                "lateness_s": np.array([row[3] for row in rows]),
            },
        )


def timing_summary(log_: RawLog) -> str:
    """One-line loop-quality summary: sample period spread and lateness (jitter feeds straight into the fit noise)."""
    period = np.diff(log_.times)
    late = log_.extras.get("lateness_s", np.zeros(1))
    return (
        f"{log_.stage}: {len(log_.times)} ticks, period {period.mean() * 1e3:.2f} ms "
        f"(std {period.std() * 1e3:.3f}, max {period.max() * 1e3:.2f}), "
        f"lateness p99 {np.percentile(late, 99) * 1e3:.2f} ms"
    )


if __name__ == "__main__":
    from autosysid.hardware.bus import MockBus
    from autosysid.hardware.clock import SimClock
    from autosysid.hardware.trajectories import build_plan
    from autosysid.mock import MINI_ARM_XML

    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex")
    calibration = ArmCalibration.symmetric(joints)
    clock = SimClock()
    bus = MockBus(calibration, clock)
    config = RecordConfig(joints=("shoulder_pan",), amplitude_scale=0.5)
    plan = build_plan(("shoulder_pan",), joints, config.control_rate_hz, 0.5, config.safety, stages=(1,))
    logs = Recorder(bus, calibration, config, clock).run(plan, xml_path=MINI_ARM_XML)
    for recorded in logs:
        print(timing_summary(recorded))
    assert len(logs) == len(plan) and not bus.torque_enabled
    print("recorder sanity checks passed")
