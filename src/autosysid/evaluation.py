"""Held-out comparison of the models of one joint, all scored with the same metric (RMS position error, ticks).

    hold_still           the joint never moves from its first measured angle            (the "do nothing" floor)
    command_as_position  the measured angle equals the command                           (ideal servo, no dynamics)
    stock_mjcf           the vendored MJCF as shipped, driven by the logged targets
    stock_plus_delay     stock MJCF with the best constant command delay (0-100 ms, chosen on the FIT logs)
    behavior_3param      delay + lag + deadband                                           (behavior.py, no gravity)
    behavior_4param      delay + lag + deadband + gravity offset                          (behavior.py)
    wrapped_mjcf         4-parameter model in front of an ideal-actuator MJCF             (sim_wrapper.py)

Errors are pooled over all held-out samples (sqrt of the mean square), not averaged per log, so a long log counts more.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from autosysid.behavior import BehaviorParams, fit_behavior, rms_error_ticks, simulate
from autosysid.data import RawLog, zero_order_hold
from autosysid.errors import DataError
from autosysid.sim_wrapper import STS3215SimWrapper, ideal_actuator_model

LOGGER = logging.getLogger("autosysid.evaluation")

CONFIG = {
    # Candidate constant delays for the stock-MJCF baseline: 0-100 ms covers every fitted Td (4-35 ms) with margin and
    # is the window mujoco.sysid searches (FitConfig.max_delay_s = 0.1).
    "stock_delay_grid_s": [round(0.01 * step, 3) for step in range(0, 11)],
}


def split_logs(logs: Sequence[RawLog], joint: str) -> tuple[list[RawLog], list[RawLog]]:
    """(fit logs, held-out logs) for `joint`.

    Held out = logs recorded with role "holdout" (stage-3 multisine) when present; otherwise the stage-2 "steps"
    segment, which is the largest-amplitude, never-fitted-on motion in the older single-stage runs.
    """
    own = [log for log in logs if joint in log.meta.get("excited", [])]
    if not own:
        raise DataError(f"no log excites '{joint}'")
    holdout = [log for log in own if log.meta.get("role") == "holdout"]
    if not holdout:
        holdout = [log for log in own if "steps" in log.stage]
    if not holdout:
        raise DataError(f"no held-out data for '{joint}': need a role='holdout' log or a 'steps' segment")
    fit = [log for log in own if not any(log is held for held in holdout)]
    if not fit:
        raise DataError(f"no fit data left for '{joint}' after holding out {[log.stage for log in holdout]}")
    return fit, holdout


@dataclass(frozen=True)
class JointEvaluation:
    joint: str
    errors_ticks: dict[str, float]
    behavior_3param: BehaviorParams
    behavior_4param: BehaviorParams
    stock_delay_s: float
    times: np.ndarray  # concatenated held-out time axis (segments laid end to end)
    measured: np.ndarray
    predictions: dict[str, np.ndarray]  # model name -> prediction on the same axis
    holdout_stages: tuple[str, ...]


def mjcf_prediction(
    model: mujoco.MjModel, params: dict[str, BehaviorParams], log: RawLog, joint: str, delay_s: float = 0.0
) -> np.ndarray:
    """Roll `model` (through `STS3215SimWrapper`) on the log's targets; return `joint`'s angle at the log times.

    Actuators with no column in the log are held at 0; the start pose is the first measured angles.
    """
    wrapper = STS3215SimWrapper(model, mujoco.MjData(model), params)
    names = wrapper.actuator_names
    relative = log.times - log.times[0]
    targets = np.zeros((len(log.times), len(names)))
    initial_qpos = np.zeros(model.nq)
    for index, name in enumerate(names):
        if name in log.joints:
            column = log.column(name)
            targets[:, index] = zero_order_hold(relative - delay_s, relative, log.target_rad[:, column])
            initial_qpos[wrapper.qpos_address[index]] = log.measured_rad[0, column]
    return wrapper.rollout(log.times, targets, initial_qpos=initial_qpos)[:, wrapper.actuator_index(joint)]


def _pooled(per_log_errors: list[tuple[np.ndarray, np.ndarray]]) -> float:
    return rms_error_ticks(
        np.concatenate([predicted for predicted, _ in per_log_errors]),
        np.concatenate([measured for _, measured in per_log_errors]),
    )


def evaluate_joint(model: mujoco.MjModel, logs: Sequence[RawLog], joint: str, q_horiz_rad: float | None = None):
    """Fit on the fit logs, score every model on the held-out logs. See the module docstring for the models."""
    fit_logs, held_logs = split_logs(logs, joint)
    ideal_model = ideal_actuator_model(model)

    fit_3 = fit_behavior(fit_logs, joint, fit_gravity=False).params
    fit_4 = fit_behavior(fit_logs, joint, fit_gravity=True, q_horiz_rad=q_horiz_rad).params

    def stock_error(delay_s: float, on: Sequence[RawLog]) -> float:
        return _pooled(
            [(mjcf_prediction(model, {}, log, joint, delay_s), log.measured_rad[:, log.column(joint)]) for log in on]
        )

    stock_delay = min(CONFIG["stock_delay_grid_s"], key=lambda delay: stock_error(delay, fit_logs))
    LOGGER.info("%s: best stock-MJCF delay on fit logs = %.0f ms", joint, stock_delay * 1e3)

    names = (
        "hold_still",
        "command_as_position",
        "stock_mjcf",
        "stock_plus_delay",
        "behavior_3param",
        "behavior_4param",
        "wrapped_mjcf",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    measured_parts, time_parts, offset = [], [], 0.0
    for log in held_logs:
        column = log.column(joint)
        measured, target = log.measured_rad[:, column], log.target_rad[:, column]
        predictions = {
            "hold_still": np.full_like(measured, measured[0]),
            "command_as_position": target,
            "stock_mjcf": mjcf_prediction(model, {}, log, joint),
            "stock_plus_delay": mjcf_prediction(model, {}, log, joint, stock_delay),
            "behavior_3param": simulate(fit_3, log.times, target, measured[0]),
            "behavior_4param": simulate(fit_4, log.times, target, measured[0]),
            "wrapped_mjcf": mjcf_prediction(ideal_model, {joint: fit_4}, log, joint),
        }
        for name, values in predictions.items():
            collected[name].append(values)
        measured_parts.append(measured)
        time_parts.append(log.times - log.times[0] + offset)
        offset += log.duration_s + 1.0

    measured_all = np.concatenate(measured_parts)
    joined = {name: np.concatenate(parts) for name, parts in collected.items()}
    return JointEvaluation(
        joint=joint,
        errors_ticks={name: rms_error_ticks(values, measured_all) for name, values in joined.items()},
        behavior_3param=fit_3,
        behavior_4param=fit_4,
        stock_delay_s=stock_delay,
        times=np.concatenate(time_parts),
        measured=measured_all,
        predictions=joined,
        holdout_stages=tuple(log.stage for log in held_logs),
    )


if __name__ == "__main__":
    from autosysid.mock import MINI_ARM_XML

    demo_params = BehaviorParams(0.016, 0.06, 0.006, 0.004)
    demo_logs = []
    for seed, stage in enumerate(("stage1_step", "stage2_chirp", "stage2_steps")):
        rng = np.random.default_rng(seed)
        t = np.arange(0, 10.0, 0.01)
        u = 0.12 * np.sin(2 * np.pi * (0.3 + 0.4 * t / 10) * t + seed)
        y = simulate(demo_params, t, u, 0.0) + rng.normal(0, 3e-4, len(t))
        demo_logs.append(RawLog(t, ("shoulder_pan",), u[:, None], y[:, None], stage, {"excited": ["shoulder_pan"]}))
    result = evaluate_joint(mujoco.MjModel.from_xml_path(str(MINI_ARM_XML)), demo_logs, "shoulder_pan")
    for name, value in result.errors_ticks.items():
        print(f"{name:20s} {value:7.2f} ticks")
    assert result.errors_ticks["behavior_4param"] < result.errors_ticks["stock_mjcf"]
    print("ok")
