"""Synthetic arm: a "truth" MuJoCo model with known dynamics, command latency, encoder quantization and noise.

Used by the tests, the demo and the examples to exercise the full pipeline without hardware, and to measure how well
the fit recovers *known* parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.rollout as mujoco_rollout
import numpy as np
from mujoco import sysid

from autosysid.data import RawLog, step_averaged_hold
from autosysid.errors import ConfigError
from autosysid.excitation import PROTOCOL_PARTS, chirp, protocol_signal
from autosysid.model import JointDynamics, apply_dynamics, load_spec, read_dynamics, sensor_name

MINI_ARM_XML = Path(__file__).parent / "assets" / "mini_arm.xml"
# "Truth" used by `autosysid demo`: multipliers on the MJCF values plus a command latency (a known answer).
DEMO_TRUTH_SCALE = {"kp": 0.7, "kv": 1.5, "damping": 1.4, "frictionloss": 1.5, "armature": 0.8}
DEMO_DELAY_S = 0.026


@dataclass(frozen=True)
class MockSettings:
    encoder_ticks: int = 4096  # STS3215 12-bit position
    noise_std_rad: float = 0.0005  # ~ a third of a tick of extra read noise on top of quantization
    timing_jitter_s: float = 0.0005  # std of control-loop timestamp jitter at 100 Hz (python loop estimate)
    seed: int = 0


def jittered_times(duration_s: float, rate_hz: float, jitter_s: float, rng: np.random.Generator) -> np.ndarray:
    """Tick timestamps at `rate_hz` with Gaussian jitter; strictly increasing, first tick exactly at 0."""
    n_ticks = int(duration_s * rate_hz) + 1
    times = np.arange(n_ticks) / rate_hz
    times[1:] += rng.normal(0.0, jitter_s, n_ticks - 1)
    if np.any(np.diff(times) <= 0):
        raise ConfigError("jitter too large relative to the control period")
    return times


class MockArm:
    """Truth model + latency + sensor imperfections. `truth`/`delays_s` are per joint; unlisted joints are nominal."""

    def __init__(
        self,
        xml_path: str | Path = MINI_ARM_XML,
        truth: dict[str, JointDynamics] | None = None,
        delays_s: dict[str, float] | None = None,
        settings: MockSettings | None = None,
    ) -> None:
        settings = settings or MockSettings()
        self.settings = settings
        self.delays_s = dict(delays_s or {})
        probe = load_spec(xml_path)
        self.joint_names = tuple(joint.name for joint in probe.joints)
        self.spec = load_spec(xml_path, sensed_joints=self.joint_names)
        self.nominal = {name: read_dynamics(self.spec, name) for name in self.joint_names}
        self.truth = {**self.nominal, **(truth or {})}
        for name, dynamics in self.truth.items():
            apply_dynamics(self.spec, name, dynamics)
        self.model = self.spec.compile()
        self._rng = np.random.default_rng(settings.seed)

    def record(self, times: np.ndarray, target_rad: np.ndarray, stage: str = "mock") -> RawLog:
        """Run the truth model on `target_rad` (N, J in `joint_names` order) and sample it at `times`."""
        times = np.asarray(times, dtype=float)
        if target_rad.shape != (len(times), len(self.joint_names)):
            raise ConfigError(f"target_rad shape {target_rad.shape} != {(len(times), len(self.joint_names))}")
        model, dt = self.model, self.model.opt.timestep
        relative = times - times[0]
        sim_times = np.arange(int(relative[-1] / dt) + 1) * dt
        ctrl = np.zeros((len(sim_times), model.nu))
        for actuator_id in range(model.nu):
            name = model.actuator(actuator_id).name
            column = self.joint_names.index(name)
            delay = self.delays_s.get(name, 0.0)
            ctrl[:, actuator_id] = step_averaged_hold(sim_times, relative + delay, target_rad[:, column])

        model_order = [self.joint_names.index(model.joint(i).name) for i in range(model.njnt)]
        initial = sysid.create_initial_state(model, target_rad[0][model_order])
        data = mujoco.MjData(model)
        state, sensordata = mujoco_rollout.rollout(model, data, initial, ctrl)
        state, sensordata = np.squeeze(state, axis=0), np.squeeze(sensordata, axis=0)

        measured = np.zeros_like(target_rad)
        for column, name in enumerate(self.joint_names):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name(name))
            address = model.sensor_adr[sensor_id]
            measured[:, column] = np.interp(relative, state[:, 0], sensordata[:, address])
        tick = 2 * np.pi / self.settings.encoder_ticks
        measured = np.round(measured / tick) * tick + self._rng.normal(0.0, self.settings.noise_std_rad, measured.shape)
        return RawLog(
            times=times,
            joints=self.joint_names,
            target_rad=target_rad,
            measured_rad=measured,
            stage=stage,
            meta={"mock": True, "delays_s": self.delays_s, "truth": {k: v.to_dict() for k, v in self.truth.items()}},
        )


def record_protocol(
    arm: MockArm, joint: str, rng: np.random.Generator, parts: str = "rcs", rate_hz: float = 100.0, scale: float = 1.0
) -> list[RawLog]:
    """Three-part single-joint recipe: r = slow ramps, c = unsaturated chirp, s = slew-limited saturating steps.

    Each part exposes different parameters (docs/data-contract.md section 4): ramps -> friction/bias, chirp -> delay and
    tracking dynamics, steps -> kv (speed plateau) and armature (initial acceleration). Other joints hold at zero.
    """
    column = arm.joint_names.index(joint)
    logs = []
    for part in parts:
        stage, duration_s = PROTOCOL_PARTS[part]
        times = jittered_times(duration_s * scale, rate_hz, arm.settings.timing_jitter_s, rng)
        targets = np.zeros((len(times), len(arm.joint_names)))
        targets[:, column] = protocol_signal(part, times)
        logs.append(arm.record(times, targets, stage=stage))
    return logs


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    arm = MockArm(delays_s={"shoulder_pan": 0.026})
    times = jittered_times(10.0, 100.0, 0.0005, rng)
    targets = np.zeros((len(times), len(arm.joint_names)))
    targets[:, 0] = chirp(times, 0.2, 4.0, 0.3)
    log = arm.record(times, targets)
    print(
        f"recorded {len(log.times)} ticks over {log.duration_s:.2f} s; pan range "
        f"[{log.measured_rad[:, 0].min():.3f}, {log.measured_rad[:, 0].max():.3f}] rad"
    )
    assert np.max(np.abs(log.measured_rad[:, 0])) > 0.2, "arm should follow the chirp"
    print("mock sanity checks passed")
