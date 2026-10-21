import signal

import numpy as np
import pytest

from autosysid.config import SafetyLimits
from autosysid.errors import HardwareError, SafetyViolation
from autosysid.hardware.bus import Fault, MockBus, ServoSnapshot
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import SimClock
from autosysid.hardware.safety import GoalLimiter, SafetyMonitor, SafetySession, safe_stop

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
RATE_HZ = 100.0


@pytest.fixture
def calibration():
    return ArmCalibration.symmetric(JOINTS)


def snapshot(ticks=(2047, 2047, 2047), temperature=25.0, voltage=12.0) -> ServoSnapshot:
    n = len(ticks)
    return ServoSnapshot(
        0.0, 0.001, np.array(ticks), np.zeros(n, int), np.zeros(n, int), np.full(n, voltage), np.full(n, temperature)
    )


# ---- GoalLimiter ---------------------------------------------------------------------------------------------


def make_limiter(limits=None, half_range=1.0):
    limits = limits or SafetyLimits()
    return GoalLimiter(limits, np.full(1, -half_range), np.full(1, half_range), np.zeros(1), 0.01)


def test_limiter_enforces_position_rate_and_acceleration():
    limits, limiter = SafetyLimits(), make_limiter()
    goals = np.array([limiter.step(np.array([5.0]))[0] for _ in range(400)])
    velocity = np.diff(goals) / 0.01
    assert goals.max() <= 1.0 + 1e-12
    assert np.max(np.abs(velocity)) <= limits.max_goal_rate_rad_s + 1e-9
    assert np.max(np.abs(np.diff(velocity))) / 0.01 <= limits.max_goal_accel_rad_s2 + 1e-6


def test_limiter_does_not_overshoot_a_stopped_target():
    limiter = make_limiter(half_range=2.0)
    goals = np.array([limiter.step(np.array([1.0]))[0] for _ in range(400)])
    assert goals.max() <= 1.0 + 0.003  # braking-aware: no overshoot beyond a tick or so


def test_limiter_is_transparent_for_an_admissible_trajectory():
    limiter = make_limiter()
    times = np.arange(0, 6, 0.01)
    desired = 0.3 * np.sin(2 * np.pi * 0.5 * times) * np.minimum(times, 1.0)  # gentle, within rate and accel limits
    out = np.array([limiter.step(np.array([x]))[0] for x in desired])
    assert np.max(np.abs(out - desired)) < 0.002 and limiter.n_limited == 0


def test_limiter_caps_dt_after_a_stalled_tick():
    limiter = make_limiter()
    jump = limiter.step(np.array([1.0]), dt=5.0)[0]  # a 5 s gap must not allow a 5 s worth of motion
    assert jump < 0.1


# ---- SafetyMonitor -------------------------------------------------------------------------------------------


def make_monitor(calibration, **limits):
    return SafetyMonitor(SafetyLimits(**limits), JOINTS, calibration)


def test_monitor_accepts_a_healthy_state(calibration):
    make_monitor(calibration).check_state(snapshot(), np.zeros(3), np.zeros(3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 70.0}, "over-temperature"),
        ({"voltage": 8.0}, "voltage"),
    ],
)
def test_monitor_trips_on_servo_health(calibration, kwargs, message):
    monitor = make_monitor(calibration)
    with pytest.raises(SafetyViolation, match=message):
        for _ in range(SafetyLimits().slow_fault_samples):
            monitor.check_state(snapshot(**kwargs), np.zeros(3), None)


@pytest.mark.parametrize("kwargs", [{"temperature": 64.0}, {"voltage": 9.0}])
def test_monitor_ignores_an_isolated_corrupt_servo_health_read(calibration, kwargs):
    # Phase 6: ~0.3% of real reads carry a corrupt temperature/voltage byte (64 C on a 23 C servo)
    monitor = make_monitor(calibration)
    for _ in range(50):
        monitor.check_state(snapshot(), np.zeros(3), None)
        monitor.check_state(snapshot(**kwargs), np.zeros(3), None)


def test_monitor_trips_on_tracking_error_and_range(calibration):
    monitor = make_monitor(calibration)
    with pytest.raises(SafetyViolation, match="tracking error"):
        monitor.check_state(snapshot(), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.8, 0.0]))
    with pytest.raises(SafetyViolation, match="outside its calibrated range"):
        monitor.check_state(snapshot(), np.array([0.0, 0.0, 3.0]), None)


def test_monitor_trips_on_late_ticks_and_sustained_comm_loss(calibration):
    monitor = make_monitor(calibration)
    monitor.check_tick(0.001)
    with pytest.raises(SafetyViolation, match="late"):
        monitor.check_tick(0.05)
    monitor.note_comm_failure(10.00)
    monitor.note_comm_failure(10.05)  # a short glitch is tolerated
    monitor.check_state(snapshot(), np.zeros(3), None)  # recovery resets the streak
    monitor.note_comm_failure(11.0)
    with pytest.raises(SafetyViolation, match="no valid servo response"):
        monitor.note_comm_failure(11.2)


# ---- safe stop and session -----------------------------------------------------------------------------------


def test_safe_stop_parks_slowly_then_releases(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock)
    bus.set_torque(True)
    bus.write_goal_ticks(calibration.rad_to_ticks(np.array([0.4, 0.0, 0.0])))
    clock.sleep_until(clock.now() + 1.0)
    began = clock.now()
    released = safe_stop(bus, calibration, SafetyLimits(), clock, np.zeros(3), RATE_HZ)
    assert released and not bus.torque_enabled
    assert clock.now() - began > 0.4 / 0.3 - 0.2, "must return at the slow safe-stop rate, not instantly"
    assert abs(calibration.ticks_to_rad(bus.read_state().position_ticks)[0]) < 0.02, "must end parked at the rest pose"


def test_safe_stop_with_lost_comms_leaves_torque_on(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock, faults=(Fault("comm_dropout", start_s=0.0),))
    bus.torque_enabled = True
    assert safe_stop(bus, calibration, SafetyLimits(), clock, np.zeros(3), RATE_HZ) is False
    assert bus.torque_enabled, "never drop the arm when we cannot talk to it"


def test_session_sets_and_restores_registers_and_parks(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock)
    limits = SafetyLimits()
    with SafetySession(bus, calibration, limits, RATE_HZ, clock) as session:
        assert bus.torque_enabled
        assert bus.read_setting("torque_limit").tolist() == [700, 700, 700]
        assert session.start_pose_rad.shape == (3,)
    assert not bus.torque_enabled
    assert bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]


def test_session_restores_even_when_the_body_raises(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock)
    with pytest.raises(RuntimeError, match="boom"):
        with SafetySession(bus, calibration, SafetyLimits(), RATE_HZ, clock):
            raise RuntimeError("boom")
    assert not bus.torque_enabled and bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]


def test_session_refuses_to_start_outside_the_soft_limits(calibration):
    clock = SimClock()
    bus = MockBus(
        calibration, clock, start_pose_rad=np.array([1.8, 0.0, 0.0])
    )  # inside hard range, outside the 80% band
    with pytest.raises(SafetyViolation, match="soft limits"):
        SafetySession(bus, calibration, SafetyLimits(), RATE_HZ, clock).__enter__()
    assert not bus.torque_enabled, "nothing may be engaged when the start check fails"


def test_session_refuses_to_start_when_hot(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock, faults=(Fault("overtemp", start_s=0.0, joint="elbow_flex", value=80.0),))
    with pytest.raises(SafetyViolation, match="over-temperature"):
        SafetySession(bus, calibration, SafetyLimits(), RATE_HZ, clock).__enter__()


def test_session_installs_and_restores_the_sigint_handler(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock)
    original = signal.getsignal(signal.SIGINT)
    with SafetySession(bus, calibration, SafetyLimits(), RATE_HZ, clock) as session:
        assert signal.getsignal(signal.SIGINT) == session._on_interrupt
        signal.raise_signal(signal.SIGINT)
        assert session.stop_requested, "Ctrl-C must request a safe stop, not raise mid-write"
    assert signal.getsignal(signal.SIGINT) == original


def test_session_parks_and_releases_even_if_enabling_torque_fails_halfway(calibration):
    # Phase 6: set_torque once failed on one servo after energizing the others; the session must still clean up
    clock = SimClock()
    bus = MockBus(calibration, clock)
    real_set_torque = bus.set_torque

    def flaky(enabled):
        real_set_torque(enabled)
        if enabled:
            raise HardwareError("servo 3 register 40: no status packet")

    bus.set_torque = flaky
    with pytest.raises(HardwareError):
        SafetySession(bus, calibration, SafetyLimits(), RATE_HZ, clock).__enter__()
    assert not bus.torque_enabled, "nothing may stay energized after a failed start"
    assert bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]
