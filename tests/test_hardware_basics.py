import json
import subprocess
import sys

import numpy as np
import pytest

from autosysid.errors import ConfigError, HardwareError
from autosysid.hardware.bus import Fault, MockBus
from autosysid.hardware.calibration import RAD_PER_TICK, ArmCalibration, JointCalibration
from autosysid.hardware.clock import SimClock
from autosysid.hardware.feetech import decode_sign_magnitude, port_users

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")


@pytest.fixture
def calibration():
    return ArmCalibration.symmetric(JOINTS)


def test_tick_radian_round_trip_and_scale(calibration):
    ticks = np.array([2047, 2500, 1000])
    radians = calibration.ticks_to_rad(ticks)
    assert radians[0] == 0.0 and radians[1] == pytest.approx(453 * RAD_PER_TICK)
    assert np.array_equal(calibration.rad_to_ticks(radians), ticks)


def test_drive_mode_reverses_direction():
    reversed_joint = JointCalibration("a", 1, 1, 0, 700, 3400)
    assert reversed_joint.ticks_to_rad(2147) == pytest.approx(-100 * RAD_PER_TICK)
    assert reversed_joint.rad_to_ticks(-100 * RAD_PER_TICK) == 2147


def test_rad_to_ticks_clips_to_the_encoder_range(calibration):
    assert calibration.rad_to_ticks(np.array([10.0, -10.0, 0.0])).tolist() == [4095, 0, 2047]


def test_soft_limits_are_the_central_fraction_of_the_range(calibration):
    low, high = calibration.soft_limits_rad(0.8)
    hard_low, hard_high = calibration.hard_limits_rad()
    assert np.allclose(high, 0.8 * hard_high) and np.allclose(low, 0.8 * hard_low)


def test_loads_the_lerobot_calibration_format(tmp_path):
    path = tmp_path / "cal.json"
    raw = {
        name: {"id": i + 1, "drive_mode": 0, "homing_offset": -1080, "range_min": 879, "range_max": 3255}
        for i, name in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
    }
    path.write_text(json.dumps(raw))
    calibration = ArmCalibration.from_lerobot_json(path)
    assert calibration.names[0] == "shoulder_pan" and calibration.joints["gripper"].servo_id == 6
    raw["shoulder_pan"]["id"] = 9  # wrong id for the SO-101 bus layout
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="id 9"):
        ArmCalibration.from_lerobot_json(path)
    with pytest.raises(ConfigError):
        ArmCalibration.from_lerobot_json(tmp_path / "missing.json")


def test_invalid_calibration_is_rejected():
    with pytest.raises(ConfigError):
        JointCalibration("a", 1, 2, 0, 0, 4000)
    with pytest.raises(ConfigError):
        JointCalibration("a", 1, 0, 0, 3000, 1000)


def test_sign_magnitude_decoding_matches_the_documented_register_example():
    assert decode_sign_magnitude(3128, 11) == -1080  # docs/data-contract.md: JSON -1080 <-> register 3128
    assert decode_sign_magnitude(0x8005, 15) == -5 and decode_sign_magnitude(0x0005, 15) == 5


def test_port_users_sees_another_process_but_not_ourselves(tmp_path):
    target = tmp_path / "fake_tty"
    target.write_text("")
    holder = subprocess.Popen([sys.executable, "-c", f"import time; f = open({str(target)!r}); time.sleep(30)"])
    try:
        for _ in range(100):
            users = port_users(str(target))
            if users:
                break
            subprocess.run(["sleep", "0.05"])
        assert users and users[0][0] == holder.pid
    finally:
        holder.kill()
    with open(target):
        assert port_users(str(target)) == [] or all(pid != holder.pid for pid, _ in port_users(str(target)))


# ---- MockBus ------------------------------------------------------------------------------------------------


def test_mock_bus_follows_a_goal_after_its_delay(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock, delays_s={"shoulder_pan": 0.05})
    bus.set_torque(True)
    start = clock.now()
    bus.write_goal_ticks(calibration.rad_to_ticks(np.array([0.3, 0.0, 0.0])))
    clock.sleep_until(start + 0.03)
    assert abs(calibration.ticks_to_rad(bus.read_state().position_ticks)[0]) < 0.005  # still inside the 50 ms delay
    clock.sleep_until(start + 1.0)
    assert calibration.ticks_to_rad(bus.read_state().position_ticks)[0] == pytest.approx(0.3, abs=0.01)


def test_mock_bus_torque_off_means_no_holding_force(calibration):
    clock = SimClock()
    bus = MockBus(calibration, clock)
    bus.set_torque(True)
    bus.write_goal_ticks(calibration.rad_to_ticks(np.array([0.0, 0.5, 0.0])))
    clock.sleep_until(clock.now() + 1.0)
    held = calibration.ticks_to_rad(bus.read_state().position_ticks)[1]
    bus.set_torque(False)
    clock.sleep_until(clock.now() + 1.0)
    released = calibration.ticks_to_rad(bus.read_state().position_ticks)[1]
    assert held == pytest.approx(0.5, abs=0.02) and abs(released - held) > 0.1  # gravity moves it once released


def test_mock_bus_torque_limit_is_enforced_and_restorable(calibration):
    bus = MockBus(calibration, SimClock())
    assert bus.read_setting("torque_limit").tolist() == [1000, 1000, 1000]
    bus.write_setting("torque_limit", np.array([350, 350, 350]))
    assert bus.model.actuator_forcerange[bus._act[0], 1] == pytest.approx(0.35 * 3.35)
    with pytest.raises(HardwareError, match="not allowed"):
        bus.write_setting("goal_position", np.zeros(3))


def test_mock_bus_faults(calibration):
    clock = SimClock()
    faults = (
        Fault("comm_dropout", start_s=1.0, duration_s=0.5),
        Fault("overtemp", start_s=2.0, joint="elbow_flex", value=70.0),
        Fault("undervoltage", start_s=3.0, value=9.0),
    )
    bus = MockBus(calibration, clock, faults=faults)
    origin = clock.now()
    clock.sleep_until(origin + 1.2)
    with pytest.raises(HardwareError, match="comm_dropout"):
        bus.read_state()
    clock.sleep_until(origin + 2.5)
    assert bus.read_state().temperature_c.tolist() == [25.0, 25.0, 70.0]
    clock.sleep_until(origin + 3.5)
    assert np.all(bus.read_state().voltage_v == 9.0)


def test_closed_bus_refuses_everything(calibration):
    bus = MockBus(calibration, SimClock())
    bus.close()
    with pytest.raises(HardwareError, match="closed"):
        bus.read_state()
