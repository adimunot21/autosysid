import json

import pytest

from autosysid.config import (
    ARM_JOINT_NAMES,
    JOINT_NAMES,
    SERVO_IDS,
    AutoSysIDConfig,
    FitConfig,
    RecordConfig,
    SafetyLimits,
)
from autosysid.errors import ConfigError


def test_defaults_are_valid_and_round_trip_through_json():
    config = AutoSysIDConfig()
    restored = AutoSysIDConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert isinstance(restored.record.joints, tuple)


def test_save_and_load(tmp_path):
    config = AutoSysIDConfig(record=RecordConfig(amplitude_scale=0.5, port="/dev/ttyACM1"))
    path = tmp_path / "runs" / "config.json"
    config.save(path)
    assert AutoSysIDConfig.load(path) == config


def test_servo_ids_match_bus_order():
    assert [SERVO_IDS[name] for name in JOINT_NAMES] == [1, 2, 3, 4, 5, 6]
    assert "gripper" not in ARM_JOINT_NAMES


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SafetyLimits(soft_limit_fraction=0.0),
        lambda: SafetyLimits(soft_limit_fraction=1.2),
        lambda: SafetyLimits(torque_limit_fraction=0.0),
        lambda: SafetyLimits(comm_timeout_s=-1.0),
        lambda: RecordConfig(control_rate_hz=5.0),
        lambda: RecordConfig(joints=("shoulder_pan", "not_a_joint")),
        lambda: RecordConfig(joints=()),
        lambda: RecordConfig(amplitude_scale=1.5),
        lambda: FitConfig(bound_factor=1.0),
        lambda: FitConfig(init_delay_s=0.2, max_delay_s=0.1),
        lambda: FitConfig(max_restarts=-1),
    ],
)
def test_invalid_values_are_rejected(factory):
    with pytest.raises(ConfigError):
        factory()


def test_unknown_key_is_a_config_error_not_a_type_error():
    with pytest.raises(ConfigError):
        AutoSysIDConfig.from_dict({"fit": {"max_itres": 3}})


def test_load_missing_or_corrupt_file(tmp_path):
    with pytest.raises(ConfigError):
        AutoSysIDConfig.load(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError):
        AutoSysIDConfig.load(bad)
