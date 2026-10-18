import signal

import numpy as np
import pytest

from autosysid.config import RecordConfig, SafetyLimits
from autosysid.errors import SafetyViolation
from autosysid.hardware.bus import Fault, MockBus
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import SimClock
from autosysid.hardware.recorder import Recorder, timing_summary
from autosysid.hardware.trajectories import build_plan
from autosysid.mock import MINI_ARM_XML

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
CALIBRATION = ArmCalibration.symmetric(JOINTS)


def setup(faults=(), clock=None, limits=None, scale=0.5, stages=(1,), start_pose=None):
    clock = clock or SimClock()
    config = RecordConfig(joints=("shoulder_pan",), amplitude_scale=scale, safety=limits or SafetyLimits())
    bus = MockBus(CALIBRATION, clock, faults=faults, start_pose_rad=start_pose)
    plan = build_plan(("shoulder_pan",), JOINTS, config.control_rate_hz, scale, config.safety, stages=stages)
    return bus, Recorder(bus, CALIBRATION, config, clock), plan


def test_one_log_per_segment_with_the_full_contract():
    bus, recorder, plan = setup()
    logs = recorder.run(plan, xml_path=MINI_ARM_XML)
    assert [log.stage for log in logs] == [segment.name for segment in plan]
    log = logs[0]
    assert log.joints == JOINTS and log.measured_rad.shape == log.target_rad.shape == (len(log.times), 3)
    assert np.all(np.diff(log.times) > 0) and np.diff(log.times).mean() == pytest.approx(0.01, rel=0.02)
    assert {"velocity_raw", "load_raw", "temperature_c", "voltage_v", "read_duration_s", "lateness_s"} <= set(
        log.extras
    )
    assert log.meta["excited"] == ["shoulder_pan"] and log.meta["stage"] == "stage1" and log.meta["comm_failures"] == 0
    assert "record_config" in log.meta and log.meta["limited_ticks"] == 0


def test_logged_target_is_what_was_sent_quantized_to_ticks():
    _, recorder, plan = setup()
    log = recorder.run(plan)[0]
    tick_rad = 2 * np.pi / 4096
    residual = (log.target_rad[:, 0] - CALIBRATION.joints["shoulder_pan"].ticks_to_rad(2047)) / tick_rad
    assert np.allclose(residual, np.round(residual), atol=1e-6), "targets must be whole servo ticks"


def test_measured_motion_follows_the_commanded_motion():
    _, recorder, plan = setup(scale=1.0, stages=(1,))
    log = recorder.run(plan)[1]  # ramp
    error = log.measured_rad[:, 0] - log.target_rad[:, 0]
    assert np.max(np.abs(error)) < 0.01 and np.ptp(log.target_rad[:, 0]) > 0.05


def test_held_joints_stay_at_the_start_pose():
    start = np.array([0.1, -0.2, 0.15])
    _, recorder, plan = setup(start_pose=start)
    for log in recorder.run(plan):
        assert np.allclose(log.target_rad[:, 1:], start[1:], atol=2e-3)


def test_arm_is_parked_released_and_registers_restored_after_a_run():
    bus, recorder, plan = setup()
    recorder.run(plan)
    assert not bus.torque_enabled and bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]


def test_failed_preflight_moves_nothing_and_engages_nothing():
    strict = SafetyLimits(max_tracking_error_rad=0.002)
    bus, recorder, plan = setup(limits=strict, scale=1.0, stages=(2,))
    with pytest.raises(SafetyViolation, match="pre-flight failed"):
        recorder.run(plan, xml_path=MINI_ARM_XML)
    assert not bus.torque_enabled and bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]


@pytest.mark.parametrize(
    ("fault", "message", "limits"),
    [
        (Fault("overtemp", start_s=3.0, joint="shoulder_pan", value=70.0), "over-temperature", SafetyLimits()),
        (Fault("undervoltage", start_s=3.0, value=8.0), "voltage", SafetyLimits()),
        # a blocked joint: the +-0.05 rad stage-1 steps can only open a 0.05 rad gap, so use a tighter watchdog
        (
            Fault("stuck_joint", start_s=3.0, joint="shoulder_pan"),
            "tracking error",
            SafetyLimits(max_tracking_error_rad=0.02),
        ),
    ],
)
def test_faults_trip_the_watchdogs_and_leave_the_arm_safe(fault, message, limits):
    bus, recorder, plan = setup(faults=(fault,), scale=1.0, stages=(1,), limits=limits)
    with pytest.raises(SafetyViolation, match=message):
        recorder.run(plan)
    assert not bus.torque_enabled, "after a trip the arm is parked and released"
    assert bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]


def test_short_comm_dropout_is_tolerated_and_recorded():
    bus, recorder, plan = setup(faults=(Fault("comm_dropout", start_s=3.0, duration_s=0.04),))
    log = recorder.run(plan)[0]
    assert log.meta["comm_failures"] >= 1 and len(log.times) < len(plan[0].times_s)


def test_long_comm_dropout_trips_and_leaves_torque_on():
    bus, recorder, plan = setup(faults=(Fault("comm_dropout", start_s=3.0, duration_s=5.0),))
    with pytest.raises(SafetyViolation, match="no valid servo response"):
        recorder.run(plan)
    assert bus.torque_enabled, "with the bus dead the servos keep holding; we must not claim to have released them"


class StallingClock(SimClock):
    """A control loop that hangs once (GC pause, USB hiccup): the next wake-up is 60 ms late."""

    def __init__(self, stall_after_s: float = 3.0) -> None:
        super().__init__()
        self._stall_at, self._stalled = self.now() + stall_after_s, False

    def sleep_until(self, deadline_s: float) -> None:
        if not self._stalled and deadline_s > self._stall_at:
            self._stalled, deadline_s = True, deadline_s + 0.06
        super().sleep_until(deadline_s)


def test_a_stalled_loop_trips_the_jitter_watchdog():
    bus, recorder, plan = setup(clock=StallingClock())
    with pytest.raises(SafetyViolation, match="late"):
        recorder.run(plan)
    assert not bus.torque_enabled


def test_ctrl_c_during_a_run_stops_safely():
    bus, recorder, plan = setup()
    calls = []

    def progress(index, total, name):
        calls.append(name)
        if index == 1:
            signal.raise_signal(signal.SIGINT)  # operator hits Ctrl-C between segments

    with pytest.raises(SafetyViolation, match="interrupted"):
        recorder.run(plan, progress=progress)
    assert not bus.torque_enabled and len(calls) == 2


def test_timing_summary_is_informative():
    _, recorder, plan = setup()
    assert "period 10.00 ms" in timing_summary(recorder.run(plan)[0])
