"""Phase-0 smoke test: can mujoco.sysid recover SO-101 joint parameters?

Takes the real SO-ARM100 MJCF, builds a "truth" arm with perturbed parameters
plus a command->measurement delay, quantizes the output like a 12-bit encoder
and checks that the fit recovers the truth from the nominal MJCF values.
Single joint (shoulder_pan) on purpose: it is gravity-free, so it isolates the
actuator / joint-friction identification path we depend on.

    conda run -n autosysid python scripts/smoke_sysid.py --xml <scene.xml>
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import sysid

CONFIG = {
    "joint": "shoulder_pan",  # vertical axis -> gravity does not load it
    "control_rate_hz": 100.0,  # record rate we expect from the real loop (1.3 ms sync read leaves ample margin)
    "timing_jitter_s": 0.0005,  # 0.5 ms std; python loop jitter estimate
    "duration_s": 12.0,  # long enough for a 0.2->4 Hz chirp plus a settle tail
    "chirp_hz": (0.2, 4.0),  # servo bandwidth is a few Hz at this amplitude
    "chirp_amplitude_rad": 0.35,  # ~20 deg: well inside the +-110 deg range
    "encoder_ticks": 4096,  # STS3215 12-bit position
    "measurement_noise_rad": 0.0005,  # ~ half a tick; sensor noise on top of quantization
    "solver_tolerance": 1e-8,  # MuJoCo default; tightened in experiments
    "solver_iterations": 100,  # MuJoCo default
    "seed": 0,  # reproducibility
    # truth multipliers relative to the MJCF nominal values
    "truth_scale": {"kp": 0.7, "kv": 1.5, "damping": 1.4, "frictionloss": 1.5, "armature": 0.8},
    "truth_delay_s": 0.026,  # latency to recover; a whole number of 2 ms sim steps so the ZOH of the input-delayed truth matches the output-delayed fit model
    "init_delay_s": 0.010,  # optimizer start (nominal has no notion of delay)
    "max_delay_s": 0.1,  # upper bound on latency; also the span skipped at the start of each record
    "bound_factor": 10.0,  # box bounds nominal/10 .. nominal*10
}


def load_spec(xml_path: str) -> mujoco.MjSpec:
    """Load the MJCF, disable contacts (no contacts occur; saves time), add joint-position sensors."""
    spec = mujoco.MjSpec.from_file(xml_path)
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # frictionloss is a soft constraint solved iteratively; solver-tolerance noise corrupts finite differences.
    spec.option.tolerance = CONFIG["solver_tolerance"]
    spec.option.iterations = CONFIG["solver_iterations"]
    for joint in spec.joints:
        spec.add_sensor(
            name=f"{joint.name}_pos",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=joint.name,
        )
    return spec


def set_joint_dynamics(spec: mujoco.MjSpec, joint: str, kp: float, kv: float, damping: float,
                       frictionloss: float, armature: float) -> None:
    sysid.apply_pdgain(spec, joint, np.array([kp, kv]))
    set_joint_field(spec, joint, "damping", damping)
    spec.joint(joint).frictionloss = frictionloss
    spec.joint(joint).armature = armature


def set_joint_field(spec: mujoco.MjSpec, joint: str, field: str, value: float) -> None:
    """Write a scalar joint field. MuJoCo 3.16 stores `damping` as a 3-coefficient polynomial: set c0 only."""
    if field == "damping":
        coefficients = np.array(spec.joint(joint).damping)
        coefficients[0] = value
        spec.joint(joint).damping = coefficients
    else:
        setattr(spec.joint(joint), field, float(value))


def zero_order_hold(query_times: np.ndarray, record_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Value of a command log at `query_times`, held constant between ticks (what the servo actually sees)."""
    indices = np.clip(np.searchsorted(record_times, query_times, side="right") - 1, 0, len(record_times) - 1)
    return values[indices]


def simulate_truth(spec: mujoco.MjSpec, joint: str, record_times: np.ndarray, target_rad: np.ndarray,
                   delay_s: float, rng: np.random.Generator):
    """Roll out the truth model with the command delayed by `delay_s`; sample at record_times.

    Uses `mujoco.rollout` - the same engine and sensor/time semantics as sysid's fit rollouts - so any
    residual at the truth parameters is noise, not a harness mismatch.
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)
    n_steps = int(record_times[-1] / model.opt.timestep) + 1
    sim_ctrl_times = np.arange(n_steps) * model.opt.timestep
    ctrl = np.zeros((n_steps, model.nu))
    ctrl[:, actuator_id] = zero_order_hold(sim_ctrl_times - delay_s, record_times, target_rad)
    initial_state = sysid.create_initial_state(model, np.zeros(model.nq))
    state, sensordata = rollout.rollout(model, data, initial_state, ctrl)
    sim_times, sim_values = np.squeeze(state, axis=0)[:, 0], np.squeeze(sensordata, axis=0)
    measured = np.stack(
        [np.interp(record_times, sim_times, sim_values[:, column]) for column in range(sim_values.shape[1])], axis=1
    )
    tick = 2 * np.pi / CONFIG["encoder_ticks"]
    measured = np.round(measured / tick) * tick + rng.normal(0, CONFIG["measurement_noise_rad"], measured.shape)
    return model, measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, help="path to scene.xml / so101_new_calib.xml")
    parser.add_argument("--scan-delay", action="store_true", help="stage 0: 1-D delay scan before the full fit")
    parser.add_argument("--optimizer", default="mujoco", choices=["mujoco", "scipy", "scipy_parallel_fd"])
    parser.add_argument("--eps", type=float, default=None, help="finite-difference step for the Jacobian (on multipliers)")
    args = parser.parse_args()
    joint = CONFIG["joint"]
    rng = np.random.default_rng(CONFIG["seed"])

    # --- nominal values straight from the MJCF -----------------------------------------------------
    nominal_spec = load_spec(args.xml)
    nominal_model = nominal_spec.compile()
    actuator = nominal_model.actuator(joint)
    nominal = {
        "kp": float(-actuator.biasprm[1]),
        "kv": float(-actuator.biasprm[2]),
        "damping": float(nominal_model.dof_damping[nominal_model.joint(joint).dofadr[0]]),
        "frictionloss": float(nominal_model.joint(joint).frictionloss[0]),
        "armature": float(nominal_model.joint(joint).armature[0]),
    }
    print("nominal (from MJCF):", nominal)
    truth = {key: nominal[key] * CONFIG["truth_scale"][key] for key in nominal}
    print("truth   (perturbed):", truth, "delay_s =", CONFIG["truth_delay_s"])

    # --- synthetic "measured" data ---------------------------------------------------------------
    n_samples = int(CONFIG["duration_s"] * CONFIG["control_rate_hz"])
    record_times = np.arange(n_samples) / CONFIG["control_rate_hz"]
    record_times[1:] += rng.normal(0, CONFIG["timing_jitter_s"], n_samples - 1)
    record_times = np.maximum.accumulate(record_times)
    f0, f1 = CONFIG["chirp_hz"]
    phase = 2 * np.pi * (f0 * record_times + 0.5 * (f1 - f0) / CONFIG["duration_s"] * record_times**2)
    taper = np.minimum(1.0, record_times / 1.0) * np.minimum(1.0, (CONFIG["duration_s"] - record_times) / 1.0)
    target = CONFIG["chirp_amplitude_rad"] * taper * np.sin(phase)

    truth_spec = load_spec(args.xml)
    set_joint_dynamics(truth_spec, joint, **truth)
    model, measured = simulate_truth(truth_spec, joint, record_times, target, CONFIG["truth_delay_s"], rng)

    # Control goes to sysid on an exact k*dt grid (zero-order hold of the 100 Hz log) and is never resampled by
    # the library: TimeSeries.resample(target_dt) uses linspace(first, last, ceil(T/dt)+1), so any log whose last
    # stamp is not a multiple of dt gets its time axis stretched by up to one dt - a bias on every fit.
    grid_times = np.arange(int(record_times[-1] / model.opt.timestep) + 1) * model.opt.timestep
    ctrl = np.zeros((len(grid_times), model.nu))
    ctrl[:, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)] = zero_order_hold(
        grid_times, record_times, target
    )
    initial_state = sysid.create_initial_state(model, np.zeros(model.nq))
    sensordata_ts = sysid.TimeSeries.from_names(record_times, measured, model)
    control_ts = sysid.TimeSeries(grid_times, ctrl)

    # --- parameters to fit -------------------------------------------------------------------------
    fit_spec = load_spec(args.xml)
    set_joint_dynamics(fit_spec, joint, **nominal)
    factor = CONFIG["bound_factor"]
    # Dimensionless multipliers on the nominal values (all ~1) keep the Gauss-Newton problem well scaled;
    # raw values span 0.02..1000. Delay is in milliseconds for the same reason.
    params = sysid.ParameterDict()
    params.add(sysid.Parameter(
        "pdgain", nominal=[1.0, 1.0], min_value=[1 / factor, 1 / factor], max_value=[factor, factor],
        modifier=lambda s, p: sysid.apply_pdgain(s, joint, np.array([nominal["kp"], nominal["kv"]]) * p.value),
    ))
    for field in ("damping", "frictionloss", "armature"):
        params.add(sysid.Parameter(
            field, nominal=1.0, min_value=1 / factor, max_value=factor,
            modifier=lambda s, p, field=field: set_joint_field(s, joint, field, nominal[field] * p.value[0]),
        ))
    params.add(sysid.Parameter("delay_ms", nominal=CONFIG["init_delay_s"] * 1e3, min_value=0.0, max_value=CONFIG["max_delay_s"] * 1e3))

    def modify_residual(params, pred, measured_ts, model, return_pred_all, state=None, **_):
        """Delay the prediction, window the measured data to the span it covers, take weighted diff."""
        delay = float(params["delay_ms"].value[0]) * 1e-3
        # Fixed window, independent of the delay being fitted: a window that moves with the delay changes the number
        # of residual samples, which makes the cost discontinuous and breaks finite-difference Jacobians.
        measured_win = sysid.apply_delayed_ts_window(measured_ts, pred, -CONFIG["max_delay_s"], 0.0)
        pred_at_measured = sysid.apply_resample_and_delay(pred, measured_win.times, delay, predicted_data=True)
        # Only the excited joint's sensor: normalize_residual divides each column by its own measured norm, so
        # near-zero columns of held joints would be blown up and dominate the cost.
        columns = measured_win.get_indices(f"{joint}_pos")[1]
        measured_cols = measured_win.data[:, columns]
        res = sysid.weighted_diff(pred_at_measured.data[:, columns], measured_cols)
        res = sysid.normalize_residual(res, measured_cols)
        return res, pred_at_measured, measured_win

    sequences = sysid.ModelSequences("so101", fit_spec, "chirp", initial_state, control_ts, sensordata_ts)
    residual_fn = sysid.build_residual_fn(
        models_sequences=[sequences], modify_residual=modify_residual, resample_true=False
    )

    # Sanity: the residual at the truth parameters must be at the noise floor, and far below the nominal's.
    truth_params = params.copy()
    truth_params["pdgain"].value[:] = [truth["kp"] / nominal["kp"], truth["kv"] / nominal["kv"]]
    for field in ("damping", "frictionloss", "armature"):
        truth_params[field].value[:] = truth[field] / nominal[field]
    truth_params["delay_ms"].value[:] = CONFIG["truth_delay_s"] * 1e3
    for label, candidate in (("nominal", params), ("truth", truth_params)):
        residuals = residual_fn(candidate.as_vector(), candidate)[0]
        print(f"cost at {label:<8}: sum(r^2)/2 = {0.5 * float(np.sum(np.concatenate(residuals) ** 2)):.4g}")

    if args.scan_delay:
        # Stage 0: with everything else nominal, the delay that minimizes the cost is a far better start than the
        # arbitrary init - the cost is nearly flat in the other parameters until the timing is roughly right.
        scan_ms = np.arange(0.0, 60.1, 2.0)
        scan_costs = []
        for delay_ms in scan_ms:
            params["delay_ms"].value[:] = delay_ms
            scan_costs.append(0.5 * float(np.sum(np.concatenate(residual_fn(params.as_vector(), params)[0]) ** 2)))
        params["delay_ms"].value[:] = scan_ms[int(np.argmin(scan_costs))]
        print(f"delay scan -> start at {params['delay_ms'].value[0]:.0f} ms")

    start = time.perf_counter()
    opt_params, result = sysid.optimize(params, residual_fn, optimizer=args.optimizer, max_iters=60, verbose=False,
                                     **({"eps": args.eps} if args.eps and args.optimizer == "mujoco" else {}),
                                     **({"x_scale": "jac"} if args.optimizer != "mujoco" else {}))
    print(f"fit took {time.perf_counter() - start:.1f} s")

    # --- compare -------------------------------------------------------------------------------------
    fitted = {
        "kp": nominal["kp"] * opt_params["pdgain"].value[0], "kv": nominal["kv"] * opt_params["pdgain"].value[1],
        "damping": nominal["damping"] * opt_params["damping"].value[0],
        "frictionloss": nominal["frictionloss"] * opt_params["frictionloss"].value[0],
        "armature": nominal["armature"] * opt_params["armature"].value[0],
    }
    print(f"\n{'param':<14}{'nominal':>12}{'truth':>12}{'fitted':>12}{'err %':>9}")
    for key in nominal:
        print(f"{key:<14}{nominal[key]:>12.4g}{truth[key]:>12.4g}{fitted[key]:>12.4g}"
              f"{100 * (fitted[key] - truth[key]) / truth[key]:>9.1f}")
    delay_fit = opt_params["delay_ms"].value[0] * 1e-3
    print(f"{'delay_s':<14}{CONFIG['init_delay_s']:>12.4g}{CONFIG['truth_delay_s']:>12.4g}{delay_fit:>12.4g}"
          f"{100 * (delay_fit - CONFIG['truth_delay_s']) / CONFIG['truth_delay_s']:>9.1f}")

    _, intervals = sysid.calculate_intervals(residual_fn(opt_params.as_vector(), opt_params)[0], result.jac)
    print("95% CI half-widths (kp,kv,damping,friction,armature,delay):", np.array2string(intervals, precision=3))


if __name__ == "__main__":
    main()
