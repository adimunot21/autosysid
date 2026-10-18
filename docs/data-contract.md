# AutoSysID data contract (Phase 0 findings)

Every item below was observed by running `scripts/inspect_hardware.py` (read-only) or
`scripts/smoke_sysid.py` against real hardware / the real library, not assumed.

## 1. Hardware: SO-101 follower, Feetech STS3215 bus

- Port: `/dev/ttyACM1` (stable alias `/dev/so101-follower`, udev rule keyed on CH340 serial). `ttyACM0` is an unrelated device.
- Bus: 1 Mbps (`Baud_Rate` register = 0), protocol end=0, IDs 1-6 =
  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- SDK: `feetech-servo-sdk` 1.0.0 (`import scservo_sdk`). Raw register map = `lerobot.motors.feetech.tables.STS_SMS_SERIES_CONTROL_TABLE`.
- All six: model number 777 (STS3215), firmware 3.10, supply 12.0 V (`Present_Voltage`=120, units 0.1 V),
  22-28 degC, `Operating_Mode`=0 (position), servo-side `P=16 I=0 D=32`, `Acceleration`=0,
  `Torque_Limit`=1000 (gripper 500), `Overload_Torque` 80 (gripper 25), `Protection_Current` 310 (gripper 250).
- `Present_Position`: 12-bit ticks 0..4095 (2*pi/4096 = 1.534 mrad per tick). Already includes the servo-side `Homing_Offset`.
- `Homing_Offset` / `Min|Max_Position_Limit` registers equal the LeRobot calibration JSON
  (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json`: id, drive_mode, homing_offset, range_min, range_max).
  **Negative offsets are sign-magnitude on bit 11** (JSON -1080 <-> register 3128 = 2048 + 1080).
- Sync read of 6 servos x 8 B (`Present_Position..Present_Temperature`, addr 56-63): **p50 1.26 ms, p99 1.34 ms, max 1.37 ms, 0/300 failures**.
  A 100-200 Hz logging loop has ample margin.
- Not yet observed (torque was off, arm at rest): signs/encoding of `Present_Velocity`, `Present_Load`, `Present_Current` under motion.
  To check in Phase 4 before relying on them (expect sign-magnitude).
- Tick <-> radian for the MJCF: MJCF zero = mid-range, so `rad = (ticks - (range_min + range_max)/2) * 2*pi/4096`
  per joint (sign via `drive_mode`). To be verified on hardware in Stage 1 (compare `jointpos` against a hand-posed arm).
- Raw snapshot: `docs/hardware_snapshot.json`.

## 2. Model: SO-ARM100 `Simulation/SO101/` MJCF

- Source: https://github.com/TheRobotStudio/SO-ARM100 @ `a758567` (Apache-2.0), vendored unmodified at
  `~/dev/simtoreal/assets/so101/` (`scene.xml` includes `so101_new_calib.xml`; `PROVENANCE.txt` there). Not copied into this repo (16 MB meshes).
- Six `<position class="sts3215">` actuators named like the joints, `ctrlrange` = joint range (radians),
  `forcerange` +-3.35. Nominal (class `sts3215`): `kp=998.22 kv=2.731`, joint `damping=0.60 frictionloss=0.052 armature=0.028`.
- Parameters live in `<default class="sts3215">` (inherited!) - the exporter must write explicit per-joint/actuator attributes, not edit the class.
- The `kp` comment in the MJCF says it is NOT a 1:1 mapping of servo P gain (assumes P=16).
- No `<sensor>` block: sysid sensors (`jointpos`) are added to the in-memory `MjSpec` at fit time.
- Contacts are irrelevant for free-air identification: disable them (`mjDSBL_CONTACT`) in fit specs for speed.
- **MuJoCo 3.16: `MjSpec.joint().damping` is a 3-vector** (polynomial coefficients, `[c0, 0, 0]`); write element 0 only.
  `frictionloss` and `armature` are scalars. The MJCF attribute `damping="x"` is c0.

## 3. mujoco.sysid (mujoco 3.16.0, extra `mujoco[sysid]`)

Findings from `scripts/smoke_sysid.py` (synthetic truth arm on the real MJCF, 12-bit quantization, 0.5 mrad noise, 0.5 ms timestamp jitter):

1. **Never let the library resample control with `target_dt`.** `TimeSeries.resample(target_dt=dt)` uses
   `linspace(first, last, ceil(T/dt)+1)`; if the last stamp is not a multiple of `dt` the time axis is stretched by up to one `dt`
   (about 0.7 ms drift at 12 s with realistic jitter). Build control on an exact `k*dt` grid with **zero-order hold** (the servo holds the
   goal between ticks) and pass `resample_true=False`. Measured with this fix: cost at the truth parameters = noise floor (1.1e-5).
2. **The residual vector must have constant length.** A window that moves with the fitted delay changes the sample count, making the cost
   discontinuous and breaking FD Jacobians (and crashing the scipy backend). Use a fixed window that skips the first `max_delay` seconds.
3. **Delay is a signal modifier**, not an MJCF parameter: `apply_resample_and_delay(pred, measured_times, delay, predicted_data=True)` inside `modify_residual`.
4. **`normalize_residual` divides each sensor column by its own measured norm**: restrict the residual to the excited joint's columns,
   otherwise near-zero columns of held joints are blown up and dominate the cost.
5. **Fit dimensionless multipliers on the nominal values** (all ~1; delay in ms) - raw values span 0.02..1000 and stall the solver.
6. Backends: `mujoco` (LM with FD Jacobian) works; `scipy` (`trf`, `x_scale="jac"`) did not converge on this problem. Use `mujoco`.
7. **Frozen means "leave the spec alone", not "pin at this value".** `apply_param_modifiers_spec` skips frozen parameters, so a
   parameter pinned at a non-nominal value silently reverts to the MJCF value. `optimizer._build_model` applies every modifier
   and uses `frozen` only to mean "not in the optimizer vector". (Regression test: `test_frozen_parameters_are_still_applied`.)
8. **Local minima are real, and "accepted" cannot see them.** From the nominal start the solver converged (legitimately; no
   tolerance/scaling setting changed it) to points 3-6% above the global minimum that still pass a 5x-noise-floor gate. `fit_joint` therefore
   runs every start (nominal + perturbed) and keeps the lowest data cost. Staged fitting (delay/armature/damping first, then all)
   makes the nominal start land in the right basin more often (3/4 vs 1/4 seeds).
9. **Uncertainty.** `calculate_intervals` at a single-start optimum was overconfident (kp +-3% while 21% off). After multi-start the
   intervals covered the truth in 19 of 20 identified parameters. Pinning an "unidentified" parameter at nominal biases the others
   and makes their intervals conditional on the wrong pin; a log-normal prior (1 sigma = x3) keeps every curvature finite instead.
10. Solver tolerance (`mjModel.opt.tolerance`), tighter `xtol/gtol`, more iterations and `x_scale="jac"` did not change which
    minimum is found; the scipy backends were worse. Not worth tuning.

## 4. Identifiability: the excitation decides what can be fitted

Measured on the mock arm with the SO-ARM100 `sts3215` values (`kp=998`, `kv=2.73`, `forcerange=+-3.35`):

- `kv * v` alone reaches the force clamp at about 1.2 rad/s, so the *model's* top speed is about 1 rad/s, and a 0.35 rad / 4 Hz
  chirp (about 9 rad/s, 220 rad/s^2) drives the actuator **saturated 44-52% of the time with force chatter**. In that
  bang-bang regime responses are chaotic in the parameters: finite-difference Jacobian columns differed by 1000x across step sizes.
- Conversely a gentle chirp (<= 0.2 rad, <= 1.5 Hz) never saturates and tracks within 1-4 mrad, which is about one encoder
  tick (1.5 mrad), so almost nothing but delay is identifiable from it.
- **No single signal identifies everything.** Fitting three sequences jointly does: slow triangle ramps (friction, bias),
  an unsaturated chirp (delay, tracking), and slew-limited steps of 0.15-0.3 rad that briefly saturate (speed plateau -> kv,
  initial acceleration -> armature). Result on the mock arm (best of 3 starts, 4 noise seeds): kp, kv, damping, armature within
  7%, delay within 0.3 ms, cost 2.3-2.4x the quantization floor (the truth itself scores 2.3x because of the injected read noise).
- Friction (0.05 N.m against a 3.35 N.m clamp) is only weakly identified: errors of 2-25%, reported with a +-45% interval.
- Consequence for hardware (Phase 4): excitation amplitude/frequency must be sized per joint from what the arm actually tracks
  (Stage 1), and a pre-flight simulation must report the saturated fraction before any motion is commanded.
