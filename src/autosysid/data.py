"""Recorded logs and their conversion to `mujoco.sysid` inputs.

The conversion rules here are not stylistic; each one exists because the library silently biases fits otherwise
(docs/data-contract.md section 3):
  * control goes to sysid on an exact k*dt grid with zero-order hold, never through `TimeSeries.resample(target_dt)`;
  * the grid is built from the model timestep, so the log's own (jittery) timestamps only enter via the ZOH lookup.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import sysid

from autosysid.errors import DataError


@dataclass
class RawLog:
    """One recorded run. Row i is control tick i: `measured_rad[i]` was read, then `target_rad[i]` was written.

    The written goal is held by the servo until the next tick, so a goal is active on [times[i], times[i+1]).
    """

    times: np.ndarray  # (N,) seconds, strictly increasing, arbitrary origin (perf_counter on hardware)
    joints: tuple[str, ...]  # column names for the two arrays below
    target_rad: np.ndarray  # (N, J) commanded goal positions
    measured_rad: np.ndarray  # (N, J) present positions, calibration applied (MJCF joint frame)
    stage: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # per-tick diagnostics (N, ...): load, temperature, timing

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.target_rad = np.asarray(self.target_rad, dtype=float)
        self.measured_rad = np.asarray(self.measured_rad, dtype=float)
        self.joints = tuple(self.joints)
        n_samples, n_joints = len(self.times), len(self.joints)
        if self.times.ndim != 1 or n_samples < 10:
            raise DataError(f"times must be 1-D with >= 10 samples, got shape {self.times.shape}")
        for label, array in (("target_rad", self.target_rad), ("measured_rad", self.measured_rad)):
            if array.shape != (n_samples, n_joints):
                raise DataError(f"{label} has shape {array.shape}, expected {(n_samples, n_joints)}")
            if not np.all(np.isfinite(array)):
                raise DataError(f"{label} contains NaN/inf")
        if not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0):
            raise DataError("times must be finite and strictly increasing")
        if len(set(self.joints)) != n_joints:
            raise DataError(f"duplicate joint names: {self.joints}")
        self.extras = {name: np.asarray(values) for name, values in self.extras.items()}
        for name, values in self.extras.items():
            if len(values) != n_samples:
                raise DataError(f"extra '{name}' has {len(values)} rows, expected {n_samples}")

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] - self.times[0])

    def column(self, joint: str) -> int:
        try:
            return self.joints.index(joint)
        except ValueError:
            raise DataError(f"joint '{joint}' not in log columns {self.joints}") from None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=self.times,
            target_rad=self.target_rad,
            measured_rad=self.measured_rad,
            joints=np.array(self.joints),
            header=json.dumps({"stage": self.stage, "meta": self.meta}),
            **{f"extra_{name}": values for name, values in self.extras.items()},
        )

    @classmethod
    def load(cls, path: Path) -> RawLog:
        try:
            with np.load(path, allow_pickle=False) as archive:
                header = json.loads(str(archive["header"]))
                return cls(
                    times=archive["times"],
                    joints=tuple(str(name) for name in archive["joints"]),
                    target_rad=archive["target_rad"],
                    measured_rad=archive["measured_rad"],
                    stage=header["stage"],
                    meta=header["meta"],
                    extras={key[len("extra_") :]: archive[key] for key in archive.files if key.startswith("extra_")},
                )
        except (OSError, KeyError, ValueError) as error:
            raise DataError(f"cannot read log {path}: {error}") from error


def zero_order_hold(query_times: np.ndarray, tick_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Value of a command log at `query_times`: each tick's value is held until the next tick."""
    indices = np.clip(np.searchsorted(tick_times, query_times, side="right") - 1, 0, len(tick_times) - 1)
    return values[indices]


def step_averaged_hold(grid_times: np.ndarray, tick_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Zero-order-hold command averaged over each simulation step [g_j, g_j + dt] (the last point is held).

    MuJoCo applies one control value per step. Sampling the hold at the step start instead moves every command edge
    to the next grid point: a timing error uniform in [0, dt) (2 ms here) whose phase relative to the log's own clock
    is arbitrary. On real data that is a misfit that parameters then absorb (Phase 4: armature driven to its bound,
    delay +2 ms). Averaging puts the edge at its true sub-step position as a fractional control value.
    """
    ticks = np.asarray(tick_times, dtype=float)
    held = np.asarray(values, dtype=float)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(held[:-1] * np.diff(ticks))]
    )  # integral of the hold at each tick stamp

    def integral(times: np.ndarray) -> np.ndarray:
        index = np.clip(np.searchsorted(ticks, times, side="right") - 1, 0, len(ticks) - 1)
        return cumulative[index] + held[index] * (times - ticks[index])

    averaged = np.empty(len(grid_times))
    averaged[:-1] = np.diff(integral(grid_times)) / np.diff(grid_times)
    averaged[-1] = zero_order_hold(grid_times[-1:], ticks, held)[0]
    return averaged


def control_grid(model: mujoco.MjModel, log: RawLog) -> tuple[np.ndarray, np.ndarray]:
    """Control signal on the exact simulation grid: returns (grid_times (M,), ctrl (M, nu)) in model actuator order."""
    relative_times = log.times - log.times[0]
    grid_times = np.arange(int(np.floor(relative_times[-1] / model.opt.timestep + 1e-9)) + 1) * model.opt.timestep
    ctrl = np.zeros((len(grid_times), model.nu))
    for actuator_id in range(model.nu):
        name = model.actuator(actuator_id).name
        ctrl[:, actuator_id] = step_averaged_hold(grid_times, relative_times, log.target_rad[:, log.column(name)])
    return grid_times, ctrl


def initial_state(model: mujoco.MjModel, log: RawLog) -> np.ndarray:
    """Full-physics state at the first sample: measured joint angles, zero velocity."""
    names = [model.joint(i).name for i in range(model.njnt)]
    if model.nq != model.njnt:
        raise DataError("only models made of 1-DOF joints are supported (nq != njnt)")
    qpos = np.array([log.measured_rad[0, log.column(name)] for name in names])
    return sysid.create_initial_state(model, qpos)  # qpos is in model joint order (nq == njnt)


def torque_scales(logs: Sequence[RawLog]) -> dict[str, float]:
    """Per-joint Torque_Limit fraction the logs were recorded with (1.0 when unrecorded, e.g. synthetic logs)."""
    recorded = [log.meta.get("torque_scale") for log in logs]
    if any(entry != recorded[0] for entry in recorded):
        raise DataError("logs were recorded with different torque limits; fit them separately")
    return {name: float(value) for name, value in (recorded[0] or {}).items()}


def build_model_sequences(spec: mujoco.MjSpec, logs: Sequence[RawLog], joint: str) -> sysid.ModelSequences:
    """Pair `spec` with `logs` for fitting `joint`. The spec must already contain that joint's `jointpos` sensor."""
    if not logs:
        raise DataError("no logs given")
    model = spec.compile()
    initial_states, controls, sensors, names = [], [], [], []
    for index, log in enumerate(logs):
        grid_times, ctrl = control_grid(model, log)
        relative_times = log.times - log.times[0]
        measured = log.measured_rad[:, [log.column(joint)]]
        initial_states.append(initial_state(model, log))
        controls.append(sysid.TimeSeries(grid_times, ctrl))
        sensors.append(sysid.TimeSeries.from_names(relative_times, measured, model))
        names.append(f"{log.stage or 'log'}_{index}")
    return sysid.ModelSequences(f"fit_{joint}", spec, names, initial_states, controls, sensors)


if __name__ == "__main__":
    ticks = np.arange(0, 3, 0.01) + np.random.default_rng(0).normal(0, 1e-4, 300)
    ticks = np.maximum.accumulate(ticks)
    goal = np.sin(ticks)[:, None]
    log = RawLog(ticks, ("a",), goal, goal.copy(), stage="selftest", meta={"k": 1})
    held = zero_order_hold(np.array([ticks[5] - 1e-9, ticks[5], ticks[5] + 1e-9]), ticks, goal[:, 0])
    assert held[0] == goal[4, 0] and held[1] == goal[5, 0] and held[2] == goal[5, 0], "ZOH must hold until next tick"
    print("data sanity checks passed")
