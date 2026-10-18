import mujoco
import pytest

from autosysid.errors import DataError
from autosysid.mock import MINI_ARM_XML
from autosysid.model import JointDynamics, apply_dynamics, load_spec, read_dynamics, sensor_name


def test_read_resolves_default_class_values():
    # mini_arm.xml defines the values only in <default class="sts3215">; they must be visible per joint.
    dynamics = read_dynamics(load_spec(MINI_ARM_XML), "elbow_flex")
    assert dynamics == JointDynamics(kp=998.22, kv=2.731, damping=0.6, frictionloss=0.052, armature=0.028)


def test_apply_then_read_round_trip_and_other_joints_untouched():
    spec = load_spec(MINI_ARM_XML)
    before = read_dynamics(spec, "elbow_flex")
    changed = JointDynamics(kp=500.0, kv=3.0, damping=0.9, frictionloss=0.07, armature=0.02)
    apply_dynamics(spec, "shoulder_pan", changed)
    assert read_dynamics(spec, "shoulder_pan") == changed
    assert read_dynamics(spec, "elbow_flex") == before


def test_damping_write_only_touches_the_linear_coefficient():
    spec = load_spec(MINI_ARM_XML)
    spec.joint("shoulder_pan").damping = [0.6, 0.11, 0.22]
    apply_dynamics(spec, "shoulder_pan", JointDynamics(kp=900, kv=2, damping=1.0, frictionloss=0.05, armature=0.03))
    assert list(spec.joint("shoulder_pan").damping) == [1.0, 0.11, 0.22]


def test_sensors_added_for_requested_joints_only():
    model = load_spec(MINI_ARM_XML, sensed_joints=("shoulder_lift",)).compile()
    assert model.nsensor == 1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name("shoulder_lift")) >= 0


def test_contacts_disabled_by_default():
    spec = load_spec(MINI_ARM_XML)
    assert spec.option.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT


def test_missing_file_and_unknown_joint_raise_data_error(tmp_path):
    with pytest.raises(DataError):
        load_spec(tmp_path / "nope.xml")
    with pytest.raises(DataError):
        load_spec(MINI_ARM_XML, sensed_joints=("not_a_joint",))


def test_non_position_actuator_is_rejected(tmp_path):
    xml = tmp_path / "motor.xml"
    xml.write_text(
        '<mujoco><worldbody><body><joint name="j" axis="0 0 1"/><geom size="0.1"/></body></worldbody>'
        '<actuator><motor name="j" joint="j"/></actuator></mujoco>'
    )
    with pytest.raises(DataError, match="position"):
        read_dynamics(load_spec(xml), "j")


def test_negative_dynamics_rejected():
    with pytest.raises(DataError):
        JointDynamics(kp=-1, kv=1, damping=1, frictionloss=0, armature=0)
