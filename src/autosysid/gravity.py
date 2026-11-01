"""Gravity phase of a joint from the MJCF: the angle `q_horiz` at which gravity's holding torque is largest.

The behavioral model's command offset is `g0 * cos(q - q_horiz)` (behavior.py). Over one ~0.1 rad sweep `g0` and
`q_horiz` cannot be separated (cos is nearly constant), so `q_horiz` is taken from the link geometry here and only
`g0` is fitted, from sweeps recorded at several base poses (hardware/sweeps.py `gravity_poses_rad`).

Method: hold the other joints at `qpos`, sweep this joint through a full turn at zero velocity, read the generalized
bias force (gravity only at v = 0) and fit `a*cos(q) + b*sin(q)`; then `q_horiz = atan2(b, a)`.
`q_horiz` carries the sign convention of `qfrc_bias`; the fitted sign of g0 absorbs any pi flip.
For distal joints the result depends on the pose of the proximal ones, hence the `qpos` argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from autosysid.errors import ConfigError


@dataclass(frozen=True)
class GravityPhase:
    q_horiz_rad: float  # in (-pi, pi]; angle of maximum gravity holding torque (MJCF joint frame)
    torque_amplitude_nm: float  # peak holding torque over the sweep
    r_squared: float  # how sinusoidal the torque is (1.0 = exactly a*cos + b*sin)


def gravity_phase_from_mjcf(
    model: mujoco.MjModel, joint: str, qpos: np.ndarray | None = None, n_samples: int = 73
) -> GravityPhase:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if joint_id < 0:
        raise ConfigError(f"no joint '{joint}' in the model")
    if n_samples < 8:
        raise ConfigError(f"n_samples must be >= 8, got {n_samples}")
    base = np.array(model.qpos0 if qpos is None else qpos, dtype=float)
    if base.shape != (model.nq,) or not np.all(np.isfinite(base)):
        raise ConfigError(f"qpos must be finite with shape {(model.nq,)}")
    qpos_address, dof_address = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])

    data = mujoco.MjData(model)
    angles = np.linspace(-math.pi, math.pi, n_samples, endpoint=False)
    torque = np.empty(n_samples)
    for index, angle in enumerate(angles):
        data.qpos[:] = base
        data.qpos[qpos_address] = angle
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        torque[index] = data.qfrc_bias[dof_address]  # at v = 0 this is the torque that holds the pose against gravity

    design = np.stack([np.cos(angles), np.sin(angles), np.ones_like(angles)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, torque, rcond=None)
    cosine, sine, _ = coefficients
    amplitude = float(math.hypot(cosine, sine))
    total = float(np.sum((torque - torque.mean()) ** 2))
    residual = float(np.sum((torque - design @ coefficients) ** 2))
    return GravityPhase(
        q_horiz_rad=float(math.atan2(sine, cosine)) if amplitude > 1e-9 else 0.0,
        torque_amplitude_nm=amplitude,
        r_squared=1.0 - residual / total if total > 1e-12 else 0.0,
    )


if __name__ == "__main__":
    from autosysid.mock import MINI_ARM_XML

    demo_model = mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))
    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
        result = gravity_phase_from_mjcf(demo_model, name)
        degrees = math.degrees(result.q_horiz_rad)
        print(
            f"{name:14s} q_horiz {degrees:6.1f} deg peak {result.torque_amplitude_nm:.3f} N.m R2 {result.r_squared:.3f}"
        )
    assert gravity_phase_from_mjcf(demo_model, "shoulder_pan").torque_amplitude_nm < 1e-6
    print("ok")
