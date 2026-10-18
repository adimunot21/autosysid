"""Per-joint identification with `mujoco.sysid`.

Everything here exists to make the fit trustworthy on logs from real hardware (docs/data-contract.md section 3):
  * parameters are dimensionless multipliers on the MJCF nominal values (delay in ms), so all are ~1;
  * the residual is restricted to the fitted joint's sensor and uses a window that does not depend on the delay,
    so the residual vector has constant length;
  * a fit is only `accepted` if its cost is within a margin of the quantization noise floor, and parameters whose
    95% confidence interval is too wide are reported as not identified instead of being exported silently.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import sysid

from autosysid.config import FitConfig
from autosysid.data import RawLog, build_model_sequences, torque_scales
from autosysid.errors import DataError
from autosysid.logging_utils import get_logger
from autosysid.model import JointDynamics, load_spec, read_dynamics, scale_force_limits, sensor_name, set_joint_scalar

log = get_logger("optimizer")

# Layout of the optimizer vector when every parameter is free (one scalar Parameter each).
PARAMETER_NAMES: tuple[str, ...] = ("kp", "kv", "damping", "frictionloss", "armature", "delay")
DYNAMICS_NAMES: tuple[str, ...] = PARAMETER_NAMES[:-1]
_STAGE_ONE_FREE = {"damping", "armature", "delay"}  # smooth, nearly linear parameters: pin down timing + inertia first


@dataclass(frozen=True)
class JointFit:
    joint: str
    nominal: JointDynamics
    dynamics: JointDynamics  # fitted values; parameters that were not identified are reset to nominal
    delay_s: float  # command -> measurement latency (not an MJCF quantity)
    relative_ci: dict[str, float]  # 95% half-widths: multiplier units for dynamics, seconds for "delay"
    unidentifiable: tuple[str, ...]
    cost: float  # 0.5 * sum(residual^2) on normalized residuals
    noise_floor: float  # expected cost from quantization alone, if the model were perfect
    accepted: bool  # cost <= margin * noise_floor
    attempt_costs: tuple[float, ...]  # data cost of every start; the spread shows how multi-modal the problem was
    seconds: float


def _noise_model(logs: Sequence[RawLog], joint: str, config: FitConfig) -> tuple[float, int]:
    """(expected cost floor from encoder quantization, number of residual samples)."""
    tick = 2 * np.pi / config.encoder_ticks
    floor, n_samples = 0.0, 0
    for raw in logs:
        elapsed = raw.times - raw.times[0]
        measured = raw.measured_rad[elapsed >= config.max_delay_s, raw.column(joint)]
        # r_i = e_i / (||y|| / sqrt(2))  =>  E[0.5 * sum r^2] = N * var / ||y||^2   (var of uniform error = tick^2 / 12)
        floor += len(measured) * (tick**2 / 12.0) / float(np.sum(measured**2))
        n_samples += len(measured)
    return floor, n_samples


def expected_noise_floor(logs: Sequence[RawLog], joint: str, config: FitConfig) -> float:
    """Cost a perfect model would still have because of encoder quantization."""
    return _noise_model(logs, joint, config)[0]


def _build_model(params: sysid.ParameterDict, spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile with EVERY parameter applied. sysid's default skips frozen parameters (leaving the spec's value), so
    a parameter pinned at a non-nominal value would silently revert; here frozen only means "not optimized"."""
    for parameter in params.values():
        parameter.apply_modifier(spec)
    return spec.compile()


class _JointProblem:
    """Residual function + parameter factory for one joint and one or more logs."""

    def __init__(self, xml_path: Path, logs: Sequence[RawLog], joint: str, config: FitConfig) -> None:
        self.joint, self.config, self.logs = joint, config, list(logs)
        self.spec = load_spec(xml_path, sensed_joints=(joint,), disable_contacts=config.disable_contacts)
        scale_force_limits(self.spec, torque_scales(self.logs))  # the plant that produced the data, not the datasheet
        self.nominal = read_dynamics(self.spec, joint)
        sequences = build_model_sequences(self.spec, self.logs, joint)
        floor, n_samples = _noise_model(self.logs, joint, config)
        # Residual std per sample if the model were perfect (cost = 0.5 * N * sigma^2 => sigma = sqrt(2 * floor / N)).
        self._prior_weight = (
            np.sqrt(2 * floor / n_samples) / np.log(config.prior_sigma_factor)
            if (config.prior_sigma_factor > 1)
            else 0.0
        )
        self.data_residual_fn = sysid.build_residual_fn(
            models_sequences=[sequences],
            build_model=_build_model,
            modify_residual=self._modify_residual,
            resample_true=False,
        )

    def residual_fn(self, x: np.ndarray, params: sysid.ParameterDict, **kwargs):
        """Data residuals plus prior terms `w * ln(multiplier)` for the free dynamics parameters."""
        residuals, preds, records = self.data_residual_fn(x, params, **kwargs)
        if self._prior_weight == 0.0:
            return residuals, preds, records
        rows, offset = [], 0
        for name, parameter in params.items():
            if parameter.frozen:
                continue
            if name != "delay":
                rows.append(offset)
            offset += parameter.size
        if not rows:
            return residuals, preds, records
        prior = self._prior_weight * np.log(np.maximum(x[rows], 1e-12))  # works for (n,) and (n, n_fd) x
        return [*residuals, prior], preds, records

    def _modify_residual(self, params, pred, measured_ts, model, return_pred_all, state=None, **_):
        delay = float(params["delay"].value[0]) * 1e-3
        # Fixed window: skip the first max_delay seconds whatever the delay is (see module docstring).
        measured_win = sysid.apply_delayed_ts_window(measured_ts, pred, -self.config.max_delay_s, 0.0)
        pred_at_measured = sysid.apply_resample_and_delay(pred, measured_win.times, delay, predicted_data=True)
        columns = measured_win.get_indices(sensor_name(self.joint))[1]
        measured_cols = measured_win.data[:, columns]
        residual = sysid.weighted_diff(pred_at_measured.data[:, columns], measured_cols)
        return sysid.normalize_residual(residual, measured_cols), pred_at_measured, measured_win

    def make_params(self, start: dict[str, float], free: set[str]) -> sysid.ParameterDict:
        """`start` holds multipliers (and "delay" in ms); names outside `free` are frozen at their start value."""
        factor, joint, nominal = self.config.bound_factor, self.joint, self.nominal
        params = sysid.ParameterDict()
        gain_modifiers = {"kp": sysid.apply_pgain, "kv": sysid.apply_dgain}
        for field, apply_gain in gain_modifiers.items():
            gain = sysid.Parameter(
                field,
                nominal=1.0,
                min_value=1 / factor,
                max_value=factor,
                frozen=field not in free,
                modifier=lambda spec, p, field=field, apply_gain=apply_gain: apply_gain(
                    spec, joint, getattr(nominal, field) * float(p.value[0])
                ),
            )
            gain.value[:] = start[field]
            params.add(gain)
        for field in ("damping", "frictionloss", "armature"):
            scalar = sysid.Parameter(
                field,
                nominal=1.0,
                min_value=1 / factor,
                max_value=factor,
                frozen=field not in free,
                modifier=lambda spec, p, field=field: set_joint_scalar(
                    spec, joint, field, getattr(nominal, field) * p.value[0]
                ),
            )
            scalar.value[:] = start[field]
            params.add(scalar)
        delay = sysid.Parameter(
            "delay", nominal=1.0, min_value=0.0, max_value=self.config.max_delay_s * 1e3, frozen="delay" not in free
        )
        delay.value[:] = start["delay"]
        params.add(delay)
        return params

    def cost(self, params: sysid.ParameterDict) -> float:
        """Data-only cost (no prior): this is what is compared with the noise floor."""
        residuals = self.data_residual_fn(params.as_vector(), params)[0]
        return 0.5 * float(np.sum(np.concatenate(residuals) ** 2))

    def optimize(self, params: sysid.ParameterDict) -> tuple[sysid.ParameterDict, Any]:
        return sysid.optimize(
            params, self.residual_fn, optimizer=self.config.optimizer, max_iters=self.config.max_iters, verbose=False
        )


def _values(params: sysid.ParameterDict) -> dict[str, float]:
    return {
        "kp": float(params["kp"].value[0]),
        "kv": float(params["kv"].value[0]),
        "damping": float(params["damping"].value[0]),
        "frictionloss": float(params["frictionloss"].value[0]),
        "armature": float(params["armature"].value[0]),
        "delay": float(params["delay"].value[0]),
    }


def _confidence(problem: _JointProblem, params: sysid.ParameterDict, result: Any, free: set[str]) -> dict[str, float]:
    """95% half-widths for the free parameters (inf where the Jacobian is rank deficient)."""
    free_names = [name for name in PARAMETER_NAMES if name in free]
    residuals = problem.residual_fn(params.as_vector(), params)[0]
    _, intervals = sysid.calculate_intervals(residuals, result.jac)
    interval_by_name = dict(zip(free_names, intervals, strict=True))
    # report delay in seconds, dynamics as multiplier half-widths
    return {name: (width * 1e-3 if name == "delay" else float(width)) for name, width in interval_by_name.items()}


def _start_point(config: FitConfig, rng: np.random.Generator | None) -> dict[str, float]:
    if rng is None:  # attempt 0: the MJCF nominal values and the configured latency guess
        return {
            "kp": 1.0,
            "kv": 1.0,
            "damping": 1.0,
            "frictionloss": 1.0,
            "armature": 1.0,
            "delay": config.init_delay_s * 1e3,
        }
    factors = np.exp(rng.uniform(-np.log(3.0), np.log(3.0), 5))  # log-uniform in [1/3, 3] x nominal
    names = ("kp", "kv", "damping", "frictionloss", "armature")
    start = dict(zip(names, map(float, factors), strict=True))
    start["delay"] = float(rng.uniform(0.0, 0.5 * config.max_delay_s) * 1e3)
    return start


def fit_joint(xml_path: str | Path, logs: Sequence[RawLog], joint: str, config: FitConfig | None = None) -> JointFit:
    """Identify one joint's servo dynamics + latency from `logs`, retrying from perturbed starts if needed."""
    config = config or FitConfig()
    started = time.perf_counter()
    problem = _JointProblem(Path(xml_path), logs, joint, config)
    floor = expected_noise_floor(logs, joint, config)
    rng = np.random.default_rng(config.seed)
    all_names = set(PARAMETER_NAMES)

    best: tuple[float, sysid.ParameterDict, Any, set[str]] | None = None
    attempt_costs: list[float] = []
    for attempt in range(config.max_restarts + 1):
        start = _start_point(config, None if attempt == 0 else rng)
        params = problem.make_params(start, _STAGE_ONE_FREE if config.staged else all_names)
        params, result = problem.optimize(params)
        free = _STAGE_ONE_FREE if config.staged else all_names
        if config.staged:
            params = problem.make_params(_values(params), all_names)
            params, result = problem.optimize(params)
            free = all_names
        cost = problem.cost(params)
        attempt_costs.append(cost)
        log.info("%s attempt %d: cost %.3g = %.2f x floor", joint, attempt, cost, cost / floor)
        if best is None or cost < best[0]:
            best = (cost, params, result, free)
        # No early exit when "accepted": local minima a few percent above the global one also pass the noise-floor
        # gate (Phase 2 finding), so every start is run and the lowest cost wins.

    assert best is not None
    cost, params, result, free = best
    confidence = _confidence(problem, params, result, free)
    unidentifiable = tuple(
        name
        for name, width in confidence.items()
        if not np.isfinite(width) or width > (config.max_delay_ci_s if name == "delay" else config.max_relative_ci)
    )
    values = _values(params)
    dynamics_unid = [name for name in unidentifiable if name != "delay"]
    if dynamics_unid:
        # Re-estimate the rest with the unidentified parameters pinned at nominal so the exported set is self-consistent
        # (it reproduces the data). The CIs reported stay those of the FULL fit: intervals conditional on a wrong pin
        # would be overconfident (Phase 4: damping/armature "+-0%" while 3x off).
        pinned = {**values, **{name: 1.0 for name in dynamics_unid}}
        params = problem.make_params(pinned, all_names - set(dynamics_unid))
        params, result = problem.optimize(params)
        cost = problem.cost(params)
        values = _values(params)
    fitted = problem.nominal.scaled({name: values[name] for name in PARAMETER_NAMES if name != "delay"})
    return JointFit(
        joint=joint,
        nominal=problem.nominal,
        dynamics=fitted,
        delay_s=values["delay"] * 1e-3,
        relative_ci=confidence,
        unidentifiable=tuple(sorted(set(unidentifiable))),
        cost=cost,
        noise_floor=floor,
        accepted=cost <= config.noise_floor_margin * floor,
        attempt_costs=tuple(attempt_costs),
        seconds=time.perf_counter() - started,
    )


def fit_all(
    xml_path: str | Path, logs: Sequence[RawLog], joints: Sequence[str], config: FitConfig | None = None
) -> dict[str, JointFit]:
    """Fit each joint independently (decoupled problems are smaller and far better conditioned)."""
    if not joints:
        raise DataError("no joints to fit")
    return {joint: fit_joint(xml_path, logs, joint, config) for joint in joints}


@dataclass(frozen=True)
class Validation:
    """Prediction error of one joint on data that was NOT used to fit it. The fitted model is judged by how well it
    predicts unseen motion, not by how close individual parameters are to anything."""

    joint: str
    rms_nominal_rad: float  # MJCF as given
    rms_fitted_rad: float  # with fitted dynamics and delay
    n_samples: int

    @property
    def improvement(self) -> float:
        return self.rms_nominal_rad / self.rms_fitted_rad if self.rms_fitted_rad > 0 else float("inf")


def validate_joint(
    xml_path: str | Path, fit: JointFit, logs: Sequence[RawLog], config: FitConfig | None = None
) -> Validation:
    """RMS position error (radians) of `fit.joint` on held-out `logs`, for the nominal model and the fitted one.

    Both models get the same delay treatment (nominal: `config.init_delay_s`) so the comparison is about dynamics.
    """
    config = config or FitConfig()
    squared = {"nominal": 0.0, "fitted": 0.0}
    n_samples = 0
    for held_out in logs:
        problem = _JointProblem(Path(xml_path), [held_out], fit.joint, config)
        window = held_out.times - held_out.times[0] >= config.max_delay_s
        norm_scale = float(np.linalg.norm(held_out.measured_rad[window, held_out.column(fit.joint)])) / np.sqrt(2)
        candidates = {
            "nominal": {
                **{name: 1.0 for name in PARAMETER_NAMES if name != "delay"},
                "delay": config.init_delay_s * 1e3,
            },
            "fitted": {
                **{name: getattr(fit.dynamics, name) / getattr(fit.nominal, name) for name in DYNAMICS_NAMES},
                "delay": fit.delay_s * 1e3,
            },
        }
        for label, start in candidates.items():
            params = problem.make_params(start, set(PARAMETER_NAMES))
            residual = np.concatenate([np.ravel(r) for r in problem.data_residual_fn(params.as_vector(), params)[0]])
            squared[label] += float(np.sum((norm_scale * residual) ** 2))
        n_samples += int(np.sum(window))
    return Validation(
        fit.joint,
        float(np.sqrt(squared["nominal"] / n_samples)),
        float(np.sqrt(squared["fitted"] / n_samples)),
        n_samples,
    )
