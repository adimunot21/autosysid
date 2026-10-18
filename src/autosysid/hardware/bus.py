"""The servo bus contract and a physics-backed mock.

`ServoBus` is the only thing the safety layer and recorder know about the arm. `MockBus` implements it on top of the
same truth model as `mock.MockArm`, advancing MuJoCo in (virtual) real time, so the whole record -> fit pipeline
and every
failure path (comm loss, over-temperature, a blocked joint) can be tested with no hardware.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import mujoco
import numpy as np

from autosysid.errors import ConfigError, HardwareError
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import Clock, SimClock
from autosysid.mock import MINI_ARM_XML, MockArm
from autosysid.model import JointDynamics

# Servo registers we are willing to change, by name (both are volatile SRAM registers on the STS3215).
SETTINGS = ("torque_limit", "acceleration")
NOMINAL_TORQUE_LIMIT = 1000  # factory/LeRobot value (docs/hardware_snapshot.json)


@dataclass(frozen=True)
class ServoSnapshot:
    """One sync-read of every joint on the bus, in bus order. `velocity_raw`/`load_raw` are signed but their units are
    UNVERIFIED on real hardware (docs/data-contract.md); the safety layer does not rely on them."""

    t_start_s: float  # clock time when the read began / ended; the sample is taken somewhere in between
    t_end_s: float
    position_ticks: np.ndarray  # (J,) int
    velocity_raw: np.ndarray  # (J,) int
    load_raw: np.ndarray  # (J,) int
    voltage_v: np.ndarray  # (J,)
    temperature_c: np.ndarray  # (J,)

    @property
    def t_mid_s(self) -> float:
        return 0.5 * (self.t_start_s + self.t_end_s)

    @property
    def read_duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


class ServoBus(Protocol):
    joints: tuple[str, ...]

    def read_state(self) -> ServoSnapshot:
        """Raises HardwareError when the read fails."""

    def write_goal_ticks(self, ticks: np.ndarray) -> None: ...

    def set_torque(self, enabled: bool) -> None: ...

    def read_setting(self, name: str) -> np.ndarray: ...

    def write_setting(self, name: str, values: np.ndarray) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Fault:
    """A scripted failure for `MockBus`. Times are seconds since the bus was created."""

    kind: Literal["comm_dropout", "overtemp", "undervoltage", "stuck_joint"]
    start_s: float
    duration_s: float = float("inf")
    joint: str | None = None  # overtemp / stuck_joint target
    value: float = 0.0  # overtemp: degrees C; undervoltage: volts

    def active(self, elapsed_s: float) -> bool:
        return self.start_s <= elapsed_s < self.start_s + self.duration_s


class MockBus:
    """Simulated STS3215 chain. Physics = truth model with a per-joint command delay; read/write cost virtual time."""

    def __init__(
        self,
        calibration: ArmCalibration,
        clock: Clock | None = None,
        xml_path: str | Path = MINI_ARM_XML,
        truth: dict[str, JointDynamics] | None = None,
        delays_s: dict[str, float] | None = None,
        faults: tuple[Fault, ...] = (),
        start_pose_rad: np.ndarray | None = None,
        read_duration_s: float = 0.0013,  # measured sync-read p50 (docs/data-contract.md)
        write_duration_s: float = 0.0002,
        seed: int = 0,
        physics_timestep_s: float | None = None,  # None: the model's own step. See the note in `_goal_average`.
    ) -> None:
        self.clock: Clock = clock or SimClock()
        self.arm = MockArm(xml_path, truth=truth, delays_s=delays_s)
        self.joints = self.arm.joint_names
        if tuple(calibration.names) != self.joints:
            raise ConfigError(f"calibration joints {calibration.names} != model joints {self.joints}")
        self.calibration, self.faults = calibration, faults
        self.read_duration_s, self.write_duration_s = read_duration_s, write_duration_s
        self._rng = np.random.default_rng(seed)
        self._origin_s = self.clock.now()
        self.model = self.arm.model
        if physics_timestep_s is not None:
            self.model.opt.timestep = physics_timestep_s
        self.data = mujoco.MjData(self.model)
        n_joints = len(self.joints)
        self._dof = np.array([self.model.joint(name).dofadr[0] for name in self.joints])
        self._qadr = np.array([self.model.joint(name).qposadr[0] for name in self.joints])
        self._act = np.array([self.model.actuator(name).id for name in self.joints])
        self._delays = np.array([self.arm.delays_s.get(name, 0.0) for name in self.joints])
        self._base_gain, self._base_bias = self.model.actuator_gainprm.copy(), self.model.actuator_biasprm.copy()
        self._base_forcerange = self.model.actuator_forcerange.copy()
        self._settings = {
            "torque_limit": np.full(n_joints, NOMINAL_TORQUE_LIMIT),
            "acceleration": np.zeros(n_joints, int),
        }
        start = np.zeros(n_joints) if start_pose_rad is None else np.asarray(start_pose_rad, dtype=float)
        self.data.qpos[self._qadr] = start
        mujoco.mj_forward(self.model, self.data)
        self._goal_times = [self._origin_s - 1e3]  # history of (time, goal) for the delayed zero-order-hold command
        self._goal_history = [start.copy()]
        self._sim_time = self._origin_s
        self._previous_qpos = start.copy()
        self._stuck_qpos: dict[str, float] = {}
        self.torque_enabled = False
        self._apply_torque_state()
        self.closed = False

    # ---- ServoBus --------------------------------------------------------------------------------------------

    def read_state(self) -> ServoSnapshot:
        self._check_open()
        started = self.clock.now()
        self._raise_if_dropout(started)
        sample_time = started + 0.5 * self.read_duration_s
        # Step one dt past the sample so it lies INSIDE the last simulated step and can be interpolated; stopping at
        # the last step before it would return a state up to one step stale (Phase 4: +1 ms delay, phase noise).
        # Exact as long as the command delay exceeds one step, since only commands written later could differ.
        dt = self.model.opt.timestep
        self._advance_to(sample_time + dt)
        fraction = (sample_time - (self._sim_time - dt)) / dt
        qpos = self._previous_qpos + np.clip(fraction, 0.0, 1.0) * (self.data.qpos[self._qadr] - self._previous_qpos)
        self.clock.elapse(self.read_duration_s)
        ended = self.clock.now()
        self._raise_if_dropout(ended)
        noise = self._rng.normal(0.0, 0.3, len(self.joints))  # ~0.3 tick of read noise on top of quantization
        ticks = np.clip(np.rint(self.calibration.rad_to_ticks(qpos) + noise), 0, 4095).astype(int)
        elapsed = sample_time - self._origin_s
        temperature = np.full(len(self.joints), 25.0)
        voltage = np.full(len(self.joints), 12.0)
        for fault in self.faults:
            if fault.active(elapsed) and fault.kind == "overtemp":
                temperature[self.joints.index(fault.joint)] = fault.value
            if fault.active(elapsed) and fault.kind == "undervoltage":
                voltage[:] = fault.value
        velocity = (self.data.qvel[self._dof] / (2 * np.pi / 4096)).astype(int)  # ticks/s, signed
        force = self.data.actuator_force[self._act]
        load = (1000.0 * force / np.maximum(self.model.actuator_forcerange[self._act, 1], 1e-9)).astype(int)
        return ServoSnapshot(started, ended, ticks, velocity, load, voltage, temperature)

    def write_goal_ticks(self, ticks: np.ndarray) -> None:
        self._check_open()
        self._raise_if_dropout(self.clock.now())
        goal = self.calibration.ticks_to_rad(np.asarray(ticks))
        self._goal_times.append(self.clock.now())
        self._goal_history.append(goal)
        self.clock.elapse(self.write_duration_s)

    def set_torque(self, enabled: bool) -> None:
        self._check_open()
        self._advance_to(self.clock.now())
        self.torque_enabled = enabled
        self._apply_torque_state()

    def read_setting(self, name: str) -> np.ndarray:
        self._check_open()
        return self._settings[self._setting(name)].copy()

    def write_setting(self, name: str, values: np.ndarray) -> None:
        self._check_open()
        self._advance_to(self.clock.now())
        self._settings[self._setting(name)] = np.asarray(values, dtype=int).copy()
        self._apply_torque_state()

    def close(self) -> None:
        self.closed = True

    # ---- simulation ------------------------------------------------------------------------------------------

    def _setting(self, name: str) -> str:
        if name not in SETTINGS:
            raise HardwareError(f"setting '{name}' not allowed; valid: {SETTINGS}")
        return name

    def _check_open(self) -> None:
        if self.closed:
            raise HardwareError("bus is closed")

    def _raise_if_dropout(self, now_s: float) -> None:
        elapsed = now_s - self._origin_s
        if any(f.kind == "comm_dropout" and f.active(elapsed) for f in self.faults):
            raise HardwareError("mock: no response from servos (comm_dropout fault)")

    def _apply_torque_state(self) -> None:
        """Torque off => no actuator force; torque limit scales the force clamp (STS3215 Torque_Limit is 0..1000)."""
        scale = 1.0 if self.torque_enabled else 0.0
        self.model.actuator_gainprm[:] = self._base_gain * scale
        self.model.actuator_biasprm[:] = self._base_bias * scale
        limit = self._settings["torque_limit"] / NOMINAL_TORQUE_LIMIT
        self.model.actuator_forcerange[self._act] = self._base_forcerange[self._act] * limit[:, None]

    def _goal_at(self, time_s: float) -> np.ndarray:
        """Command the servo acts on at `time_s`: latest write at least `delay` earlier (zero-order hold)."""
        goal = np.empty(len(self.joints))
        for column, delay in enumerate(self._delays):
            index = max(bisect.bisect_right(self._goal_times, time_s - delay) - 1, 0)
            goal[column] = self._goal_history[index][column]
        return goal

    def _goal_average(self, step_start_s: float, dt: float) -> np.ndarray:
        """Delayed command averaged over one physics step [start, start + dt].

        Sampling the hold at the step start would shift every command edge to the next step boundary (mean +dt/2).
        The fit (`data.step_averaged_hold`) places edges at their true sub-step position, so the plant must too. The
        plant and the fit model share one step size on purpose: with a finer plant the fit's integrator error is
        absorbed by the parameters (e.g. armature), which is right for "make THIS simulation match" but is not
        recoverable as the plant's "true" values.
        """
        times = np.asarray(self._goal_times)
        goal = np.empty(len(self.joints))
        for column, delay in enumerate(self._delays):
            start, end = step_start_s - delay, step_start_s - delay + dt
            first = max(int(np.searchsorted(times, start, side="right")) - 1, 0)
            total = 0.0
            for index in range(first, len(times)):
                begin = max(start, times[index])
                finish = min(end, times[index + 1]) if index + 1 < len(times) else end
                if begin >= end:
                    break
                total += self._goal_history[index][column] * max(finish - begin, 0.0)
            goal[column] = total / dt
        return goal

    def _advance_to(self, time_s: float) -> None:
        dt = self.model.opt.timestep
        while self._sim_time + dt <= time_s:
            elapsed = self._sim_time - self._origin_s
            self.data.ctrl[self._act] = self._goal_average(self._sim_time, dt)
            self._previous_qpos = self.data.qpos[self._qadr].copy()
            mujoco.mj_step(self.model, self.data)
            self._sim_time += dt
            for fault in self.faults:
                if fault.kind == "stuck_joint" and fault.active(elapsed):
                    index = self.joints.index(fault.joint)
                    pinned = self._stuck_qpos.setdefault(fault.joint, float(self._previous_qpos[index]))
                    self.data.qpos[self._qadr[index]] = pinned
                    self.data.qvel[self._dof[index]] = 0.0
        while len(self._goal_times) > 2 and self._goal_times[1] < self._sim_time - 1.0:  # trim history older than 1 s
            del self._goal_times[0], self._goal_history[0]


if __name__ == "__main__":
    calibration = ArmCalibration.symmetric(("shoulder_pan", "shoulder_lift", "elbow_flex"))
    bus = MockBus(calibration, delays_s={"shoulder_pan": 0.02})
    first = bus.read_state()
    assert abs(first.read_duration_s - 0.0013) < 1e-9 and first.position_ticks.shape == (3,)
    bus.set_torque(True)
    bus.write_goal_ticks(calibration.rad_to_ticks(np.array([0.3, 0.0, 0.0])))
    bus.clock.sleep_until(bus.clock.now() + 1.0)
    final = bus.read_state()
    moved = float(calibration.ticks_to_rad(final.position_ticks)[0])
    assert abs(moved - 0.3) < 0.02, f"pan should have reached ~0.3 rad, got {moved:.3f}"
    print(f"mock bus sanity checks passed (pan reached {moved:.3f} rad)")
