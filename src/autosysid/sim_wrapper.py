"""`STS3215SimWrapper`: put the fitted servo behavior (delay, lag, deadband, gravity offset) in front of MuJoCo.

A MuJoCo `<position>` actuator is an ideal, instantaneous PD law: it has no transport delay, no 60 ms lag and no
stiction deadband, which is why the stock MJCF tracks a real STS3215 poorly (docs/POSTMORTEM.md). Those effects live in
the *command path*, so the wrapper intercepts the target before `mj_step`:

    target --> BehaviorFilter (delay -> +g(q) -> deadband -> lag) --> data.ctrl --> mj_step

Design notes
  * One `BehaviorFilter` per managed joint, the same class the fitter optimises (`behavior.simulate`), so what is
    fitted is exactly what is simulated.
  * The fitted tau/deadband already contain the real servo's actuator dynamics, so the MuJoCo actuator must not add its
    own: the stock one has a +8 ms rise-time error at small signals and saturates (`forcerange` 3.35 Nm) on a 0.5 rad
    step where the real servo never did (108 ticks of error). Pass the model through `ideal_actuator_model` first;
    it returns a copy with a stiffer, higher-force PD (0.23 ticks RMS from the pure model on a chirp).
  * The filter's gravity term is fed the true `qpos`, so `g0*cos(q - q_horiz)` follows the simulated configuration.
    MuJoCo's own gravity sag at kp ~ 1000 is ~1 mrad and is not subtracted: it is inside the fitted offset's error.
  * Joints without parameters pass straight through: with an empty `params` the wrapper is exactly stock MuJoCo.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np

from autosysid.behavior import BehaviorFilter, BehaviorParams
from autosysid.data import zero_order_hold
from autosysid.errors import ConfigError, DataError


def ideal_actuator_model(
    model: mujoco.MjModel,
    stiffness_scale: float = 10.0,
    damping_scale: float = math.sqrt(10.0),
    force_scale: float = 100.0,
) -> mujoco.MjModel:
    """Copy of `model` whose position actuators are close to ideal, for use under `STS3215SimWrapper`.

    kp x10 and kv x sqrt(10) keep the damping ratio roughly constant while moving the actuator pole well above the
    fitted ~60 ms lag; forcerange x100 removes saturation the real servo never showed (docs/POSTMORTEM.md: peak
    acceleration ~20 rad/s^2 against 120-150 in the model). kp x100 was tried and is unstable at dt = 2 ms.
    The original model is not modified.
    """
    for label, value in (
        ("stiffness_scale", stiffness_scale),
        ("damping_scale", damping_scale),
        ("force_scale", force_scale),
    ):
        if not (math.isfinite(value) and value > 0):
            raise ConfigError(f"{label} must be finite and > 0, got {value}")
    ideal = copy.copy(model)
    for index in range(ideal.nu):
        is_position_servo = (
            ideal.actuator_gaintype[index] == mujoco.mjtGain.mjGAIN_FIXED
            and ideal.actuator_biastype[index] == mujoco.mjtBias.mjBIAS_AFFINE
            and ideal.actuator_gainprm[index, 0] > 0
            and np.isclose(ideal.actuator_gainprm[index, 0], -ideal.actuator_biasprm[index, 1])
        )
        if not is_position_servo:
            raise ConfigError(f"actuator '{ideal.actuator(index).name}' is not a <position> actuator")
        ideal.actuator_gainprm[index, 0] *= stiffness_scale
        ideal.actuator_biasprm[index, 1] *= stiffness_scale
        ideal.actuator_biasprm[index, 2] *= damping_scale
        if ideal.actuator_forcelimited[index]:
            ideal.actuator_forcerange[index] *= force_scale
    return ideal


class STS3215SimWrapper:
    """Drives a MuJoCo model whose actuators are STS3215 position servos, with per-joint behavioral dynamics.

    Targets are radians in actuator order, shape `(model.nu,)` (the same order as `data.ctrl`).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, params: Mapping[str, BehaviorParams]) -> None:
        self.model = model
        self.data = data
        self.params: dict[str, BehaviorParams] = dict(params)
        self.dt = float(model.opt.timestep)
        self._qpos_address = np.array(
            [model.jnt_qposadr[model.actuator_trnid[index, 0]] for index in range(model.nu)], dtype=int
        )
        self._names = [model.actuator(index).name for index in range(model.nu)]
        unknown = sorted(set(self.params) - set(self._names))
        if unknown:
            raise ConfigError(f"behavior parameters for joints not in the model: {unknown}; model has {self._names}")
        self._filters: dict[int, BehaviorFilter] = {}
        self.reset()

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return tuple(self._names)

    @property
    def qpos_address(self) -> np.ndarray:
        """`data.qpos` index of the joint driven by each actuator, in actuator order."""
        return self._qpos_address.copy()

    # -- lookup ------------------------------------------------------------------------------------------------
    def actuator_index(self, name: str) -> int:
        try:
            return self._names.index(name)
        except ValueError:
            raise ConfigError(f"no actuator '{name}'; model has {self._names}") from None

    def joint_position(self, name: str) -> float:
        return float(self.data.qpos[self._qpos_address[self.actuator_index(name)]])

    # -- simulation --------------------------------------------------------------------------------------------
    def reset(self, qpos: np.ndarray | None = None, initial_target: np.ndarray | None = None) -> None:
        """Reset the physics to `qpos` (default: the model's reference pose) and every filter to rest at that pose.

        `initial_target` (actuator order) is the command assumed to have been held before t=0; default is the
        command that holds each joint at its current angle.
        """
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            qpos = np.asarray(qpos, dtype=float)
            if qpos.shape != (self.model.nq,) or not np.all(np.isfinite(qpos)):
                raise DataError(f"qpos must be finite with shape {(self.model.nq,)}, got {qpos.shape}")
            self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self._filters = {}
        for name, params in self.params.items():
            index = self.actuator_index(name)
            q0 = float(self.data.qpos[self._qpos_address[index]])
            held = None if initial_target is None else float(initial_target[index])
            self._filters[index] = BehaviorFilter(params, dt=self.dt, q0=q0, initial_command=held)

    def step(self, target_rad: np.ndarray) -> None:
        """Advance one `model.opt.timestep` with the given actuator targets (rad)."""
        target = np.asarray(target_rad, dtype=float)
        if target.shape != (self.model.nu,):
            raise DataError(f"target must have shape {(self.model.nu,)}, got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise DataError(f"target contains NaN/inf: {target}")
        for index in range(self.model.nu):
            behavior_filter = self._filters.get(index)
            if behavior_filter is None:
                self.data.ctrl[index] = target[index]
            else:
                measured = float(self.data.qpos[self._qpos_address[index]])
                self.data.ctrl[index] = behavior_filter.step(float(target[index]), q=measured)
        mujoco.mj_step(self.model, self.data)

    def rollout(self, times: np.ndarray, targets: np.ndarray, initial_qpos: np.ndarray | None = None) -> np.ndarray:
        """Simulate a recorded command log; return joint angles `(N, nu)` interpolated back to `times`.

        Targets are zero-order-held between log ticks. The run starts from `initial_qpos` (default reference pose)
        with every filter at rest holding `targets[0]`.
        """
        times = np.asarray(times, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if times.ndim != 1 or len(times) < 2 or targets.shape != (len(times), self.model.nu):
            raise DataError(f"times {times.shape} / targets {targets.shape} must be (N,) and (N, {self.model.nu})")
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(targets))) or np.any(np.diff(times) <= 0):
            raise DataError("times must be finite and strictly increasing; targets finite")

        relative = times - times[0]
        n_steps = int(math.ceil(relative[-1] / self.dt - 1e-9))
        grid = np.arange(n_steps) * self.dt
        held = np.stack([zero_order_hold(grid, relative, targets[:, index]) for index in range(self.model.nu)], axis=1)

        self.reset(qpos=initial_qpos, initial_target=targets[0])
        trajectory = np.empty((n_steps + 1, self.model.nu))
        trajectory[0] = self.data.qpos[self._qpos_address]
        for step_index in range(n_steps):
            self.step(held[step_index])
            trajectory[step_index + 1] = self.data.qpos[self._qpos_address]
        state_times = np.arange(n_steps + 1) * self.dt
        return np.stack(
            [np.interp(relative, state_times, trajectory[:, index]) for index in range(self.model.nu)], axis=1
        )

    # -- persistence -------------------------------------------------------------------------------------------
    @staticmethod
    def save_params(path: Path, params: Mapping[str, BehaviorParams]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: _params_to_dict(value) for name, value in params.items()}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_json(cls, model: mujoco.MjModel, data: mujoco.MjData, path: Path) -> STS3215SimWrapper:
        try:
            raw = json.loads(Path(path).read_text())
            params = {name: BehaviorParams(**values) for name, values in raw.items()}
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise ConfigError(f"cannot read behavior parameter file {path}: {error}") from error
        return cls(model, data, params)


def _params_to_dict(params: BehaviorParams) -> dict[str, float | None]:
    return {
        "delay_s": params.delay_s,
        "tau_s": params.tau_s,
        "deadband_rad": params.deadband_rad,
        "gravity_rad": params.gravity_rad,
        "q_horiz_rad": params.q_horiz_rad,
        "deadband_gravity_rad": params.deadband_gravity_rad,
    }


if __name__ == "__main__":
    from autosysid.mock import MINI_ARM_XML

    demo_model = ideal_actuator_model(mujoco.MjModel.from_xml_path(str(MINI_ARM_XML)))
    wrapped = STS3215SimWrapper(
        demo_model, mujoco.MjData(demo_model), {"shoulder_pan": BehaviorParams(0.015, 0.06, 0.006)}
    )
    demo_times = np.arange(0, 1.0, 0.01)
    demo_targets = np.zeros((len(demo_times), demo_model.nu))
    demo_targets[20:, 0] = 0.3  # step at 0.2 s
    response = wrapped.rollout(demo_times, demo_targets)[:, 0]
    rise_time = demo_times[int(np.argmax(response >= 0.3 * (1 - math.exp(-1))))] - 0.2
    print(f"step response: 63% rise at {rise_time * 1e3:.0f} ms after the step (delay 15 + tau 60 -> ~75 ms)")
    assert 0.06 < rise_time < 0.09, rise_time
    print("ok")
