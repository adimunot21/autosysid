"""MJCF access layer: load a spec ready for identification, read/write per-joint servo dynamics.

Contract (docs/data-contract.md section 2): every identified joint has a hinge/slide joint and a `<position>` actuator
with the SAME name; `kp`/`kv` live in `gainprm[0]` / `-biasprm[1]`, `-biasprm[2]`; MuJoCo 3.16 stores joint `damping`
as a 3-coefficient polynomial of which only c0 is the linear damping.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from mujoco import sysid

from autosysid.errors import DataError

DYNAMICS_FIELDS: tuple[str, ...] = ("kp", "kv", "damping", "frictionloss", "armature")


@dataclass(frozen=True)
class JointDynamics:
    """Servo + joint parameters identified per joint (SI units: N·m/rad, N·m·s/rad, N·m·s/rad, N·m, kg·m²)."""

    kp: float
    kv: float
    damping: float
    frictionloss: float
    armature: float

    def __post_init__(self) -> None:
        for name in DYNAMICS_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise DataError(f"JointDynamics.{name}={value!r} must be finite and >= 0")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def scaled(self, multipliers: dict[str, float]) -> JointDynamics:
        return JointDynamics(**{name: getattr(self, name) * multipliers.get(name, 1.0) for name in DYNAMICS_FIELDS})


def sensor_name(joint: str) -> str:
    return f"{joint}_pos"


def load_spec(
    xml_path: str | Path, sensed_joints: tuple[str, ...] = (), disable_contacts: bool = True
) -> mujoco.MjSpec:
    """Load an MJCF (includes resolved) and add one `jointpos` sensor per sensed joint.

    Contacts are disabled by default: free-air identification has none and they only cost time.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise DataError(f"MJCF not found: {path}")
    try:
        spec = mujoco.MjSpec.from_file(str(path))
    except ValueError as error:
        raise DataError(f"cannot parse {path}: {error}") from error
    if disable_contacts:
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    for joint in sensed_joints:
        _require_joint_and_actuator(spec, joint)
        spec.add_sensor(
            name=sensor_name(joint),
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=joint,
        )
    return spec


def _require_joint_and_actuator(spec: mujoco.MjSpec, joint: str) -> None:
    if spec.joint(joint) is None:
        raise DataError(f"joint '{joint}' not in model; joints: {[j.name for j in spec.joints]}")
    actuator = spec.actuator(joint)
    if actuator is None:
        raise DataError(f"model has no actuator named '{joint}' (actuators: {[a.name for a in spec.actuators]})")
    if not sysid.model_modifier.is_position_actuator(actuator):
        raise DataError(f"actuator '{joint}' is not a <position> actuator (gain/bias do not match kp/kv form)")


def scale_force_limits(spec: mujoco.MjSpec, scales: dict[str, float]) -> None:
    """Multiply each named actuator's `forcerange` by `scales[name]` (the servo `Torque_Limit` as a fraction of full).

    A reduced Torque_Limit changes the PLANT the data came from (the servo saturates earlier), so a fit must use the
    same clamp. Assumes the register scales the maximum torque linearly - to be verified on hardware (Phase 6).
    """
    for name, scale in scales.items():
        actuator = spec.actuator(name)
        if actuator is None:
            raise DataError(f"no actuator '{name}' to scale")
        if not 0.0 < scale <= 1.0:
            raise DataError(f"torque scale for '{name}' must be in (0, 1], got {scale}")
        low, high = actuator.forcerange
        actuator.forcerange = [low * scale, high * scale]


def read_dynamics(spec: mujoco.MjSpec, joint: str) -> JointDynamics:
    """Effective dynamics of `joint` as the compiled model sees them (class defaults resolved)."""
    _require_joint_and_actuator(spec, joint)
    model = spec.copy().compile()
    dof = int(model.joint(joint).dofadr[0])
    actuator = model.actuator(joint)
    return JointDynamics(
        kp=float(-actuator.biasprm[1]),
        kv=float(-actuator.biasprm[2]),
        damping=float(model.dof_damping[dof]),
        frictionloss=float(model.dof_frictionloss[dof]),
        armature=float(model.dof_armature[dof]),
    )


def set_joint_scalar(spec: mujoco.MjSpec, joint: str, field: str, value: float) -> None:
    """Set one of damping / frictionloss / armature on a joint (damping: polynomial coefficient c0 only)."""
    spec_joint = spec.joint(joint)
    if field == "damping":
        coefficients = np.array(spec_joint.damping)
        coefficients[0] = value
        spec_joint.damping = coefficients
    elif field in ("frictionloss", "armature"):
        setattr(spec_joint, field, float(value))
    else:
        raise DataError(f"'{field}' is not a joint scalar field")


def apply_dynamics(spec: mujoco.MjSpec, joint: str, dynamics: JointDynamics) -> None:
    """Write `dynamics` into the spec (explicit per-joint values, overriding any class default)."""
    _require_joint_and_actuator(spec, joint)
    sysid.apply_pdgain(spec, joint, np.array([dynamics.kp, dynamics.kv]))
    for field in ("damping", "frictionloss", "armature"):
        set_joint_scalar(spec, joint, field, getattr(dynamics, field))


if __name__ == "__main__":
    xml = Path(__file__).parent / "assets" / "mini_arm.xml"
    spec = load_spec(xml, sensed_joints=("shoulder_pan",))
    nominal = read_dynamics(spec, "shoulder_pan")
    print("nominal:", nominal)
    changed = JointDynamics(kp=500.0, kv=3.0, damping=0.9, frictionloss=0.07, armature=0.02)
    apply_dynamics(spec, "shoulder_pan", changed)
    assert read_dynamics(spec, "shoulder_pan") == changed, "write/read round trip failed"
    untouched = read_dynamics(load_spec(xml), "elbow_flex")
    assert read_dynamics(spec, "elbow_flex") == untouched, "other joints must not change"
    print("model sanity checks passed")
