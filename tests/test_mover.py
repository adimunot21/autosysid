"""Commanded single-pose moves: slow, limiter-gated, torque held, never releases."""

from __future__ import annotations

import numpy as np
import pytest

from autosysid.config import SafetyLimits
from autosysid.errors import ConfigError, SafetyViolation
from autosysid.hardware.bus import Fault, MockBus
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import SimClock
from autosysid.hardware.mover import MoveResult, move_to

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
CALIBRATION = ArmCalibration.symmetric(JOINTS)
TICK = 2 * np.pi / 4096


def make(start=(0.0, 0.0, 0.0), faults=()):
    clock = SimClock()
    return MockBus(CALIBRATION, clock, faults=faults, start_pose_rad=np.array(start)), clock


def run(bus, clock, targets, **kwargs):
    return move_to(bus, CALIBRATION, SafetyLimits(), clock, targets, **kwargs)


def present(bus):
    return CALIBRATION.ticks_to_rad(bus.read_state().position_ticks)


class TestMove:
    def test_reaches_the_target_and_holds_torque(self):
        bus, clock = make()
        result = run(bus, clock, {"shoulder_lift": 0.4})
        assert isinstance(result, MoveResult)
        assert present(bus)[1] == pytest.approx(0.4, abs=10 * TICK)
        assert bus.torque_enabled, "a move must never leave the arm limp"
        assert bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]

    def test_other_joints_stay_put(self):
        bus, clock = make(start=(0.1, 0.0, -0.2))
        run(bus, clock, {"shoulder_lift": 0.4})
        assert present(bus)[0] == pytest.approx(0.1, abs=5 * TICK)
        assert present(bus)[2] == pytest.approx(-0.2, abs=5 * TICK)

    def test_result_reports_start_goal_and_end(self):
        bus, clock = make(start=(0.0, 0.1, 0.0))
        result = run(bus, clock, {"shoulder_lift": 0.4})
        assert result.start_rad[1] == pytest.approx(0.1, abs=2 * TICK)
        assert result.goal_rad[1] == pytest.approx(0.4, abs=TICK)
        assert result.end_rad[1] == pytest.approx(0.4, abs=10 * TICK)
        assert result.duration_s > 0.3 / 0.3 * 0.9  # 0.3 rad at 0.3 rad/s takes about a second

    def test_never_exceeds_the_rate_cap(self):
        bus, clock = make()
        sent = []
        original = bus.write_goal_ticks
        bus.write_goal_ticks = lambda ticks: (sent.append(np.array(ticks)), original(ticks))[1]
        run(bus, clock, {"shoulder_lift": 0.5}, rate_rad_s=0.3)
        goals = np.array(sent) * TICK
        assert np.max(np.abs(np.diff(goals, axis=0))) * 100.0 <= 0.3 * 1.05 + TICK * 100

    def test_multiple_joints_in_one_move(self):
        bus, clock = make()
        run(bus, clock, {"shoulder_lift": 0.3, "elbow_flex": -0.3})
        assert present(bus)[1:] == pytest.approx([0.3, -0.3], abs=10 * TICK)


class TestRefusals:
    def test_target_outside_soft_limits_moves_nothing(self):
        bus, clock = make()
        lower, upper = CALIBRATION.soft_limits_rad(SafetyLimits().soft_limit_fraction)
        with pytest.raises(ConfigError, match="soft limit"):
            run(bus, clock, {"shoulder_lift": float(upper[1]) + 0.1})
        assert not bus.torque_enabled

    def test_unknown_joint(self):
        bus, clock = make()
        with pytest.raises(ConfigError, match="wrist_roll"):
            run(bus, clock, {"wrist_roll": 0.1})

    def test_empty_targets(self):
        bus, clock = make()
        with pytest.raises(ConfigError):
            run(bus, clock, {})

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_target(self, value):
        bus, clock = make()
        with pytest.raises(ConfigError):
            run(bus, clock, {"shoulder_lift": value})

    @pytest.mark.parametrize("rate", [0.0, -0.1, 5.0])
    def test_rate_must_be_positive_and_not_above_the_safety_cap(self, rate):
        bus, clock = make()
        with pytest.raises(ConfigError):
            run(bus, clock, {"shoulder_lift": 0.1}, rate_rad_s=rate)

    def test_start_pose_outside_soft_limits_is_refused(self):
        bus, clock = make(start=(0.0, 3.0, 0.0))
        with pytest.raises(SafetyViolation, match="outside"):
            run(bus, clock, {"shoulder_lift": 0.1})


class TestTrip:
    def test_a_fault_mid_move_returns_to_the_start_pose_and_holds(self):
        fault = Fault("overtemp", start_s=0.5, joint="elbow_flex", value=80.0)
        bus, clock = make(start=(0.0, 0.0, 0.0), faults=(fault,))
        with pytest.raises(SafetyViolation, match="over-temperature"):
            run(bus, clock, {"shoulder_lift": 0.6})
        assert bus.torque_enabled
        assert present(bus)[1] == pytest.approx(0.0, abs=0.05), "a tripped move goes back where it started"
