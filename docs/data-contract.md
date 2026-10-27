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

## 5. Export (Phase 3)

Verified against the real SO-101 model (`scene.xml` -> `<include so101_new_calib.xml>`; joints use `class="sts3215"`):

- `scene.xml` contains no joints; they are in the included file, so the exporter follows `<include>` chains (relative to the
  including file, then the main file) and re-emits only files that changed, rewriting the include path (directory part kept).
- Joint values are inherited from `<default class="sts3215">`, but actuators override `forcerange` per element (+-3.35, while the class says
  +-2.94). `forcerange` is not fitted, so it is left alone; the effective limit comes from the element, not the class.
- The upstream folder also holds a stale `joints_properties.xml` (`kp=17.8, kv=0`) that no file includes. It is not the model;
  the canonical values are in `so101_new_calib.xml` (`kp=998.22, kv=2.731`).
- Edits are located with expat byte offsets and applied to the start tags only. Exporting `elbow_flex` changes exactly 2 lines in
  the model file (the `<joint>` and the `<position>`) plus the `<include>` line in the scene; comments/ordering/quotes are untouched.
- Latency (`delay_s`) has no MJCF representation; it is written to `<out>.calibration.json` together with 95% CIs, costs and
  the fit config. The JSON uses file names, not absolute paths.
- Output must be in the same directory as the source (relative `meshdir` and includes keep working without copying 16 MB of meshes).

## 6. Hardware layer and record -> fit consistency (Phase 4)

Everything here was verified against `MockBus` (physics-backed, fault-injectable) and the `FeetechBus` wiring against a fake
`scservo_sdk`; **no port was opened and no register written**. What still has to be checked on the real arm is marked (Phase 6).

### 6.1 Bus facts the code relies on
- Sync-read block `Present_Position..Present_Temperature`, addr 56, 8 bytes: position(2) velocity(2) load(2) voltage(1, 0.1 V) temperature(1).
  Goal_Position addr 42 (2 B, little-endian), Torque_Enable 40, Acceleration 41, Torque_Limit 48 (2 B). Only `torque_limit` and
  `acceleration` (both SRAM, volatile) are writable through the bus API; EEPROM registers are unreachable by design.
- `Present_Position` is not sign-encoded. Velocity (sign bit 15) and load (sign bit 10) follow LeRobot's table; **units and signs under
  motion are unverified (Phase 6)** and no watchdog uses them. Tracking error is the stall/collision proxy.
- Tick <-> radian: LeRobot homing makes the calibration pose read 2047 ticks, and the MJCF zero is that pose, so
  `rad = (ticks - 2047) * 2*pi/4096`. (Section 1 said "range mid"; the ranges are not symmetric about 2047, e.g. pan 709..3436, and
  homing semantics are the better source.) **Verify with a hand-posed arm (Phase 6).**
- `FeetechBus.open()` refuses a port another process holds (scan of `/proc/*/fd`), pings every servo, checks model 777 and
  position mode, and writes nothing.

### 6.2 Safety behaviour (all exercised with injected faults)
- Limiter: soft limits (80% of calibrated range), rate cap, acceleration cap, braking-aware so a stopped target is not overshot.
- Watchdogs: temperature, supply voltage, calibrated-range excursion, tracking error, late tick, sustained read failure.
- A trip holds, returns to the start pose at 0.3 rad/s, then releases torque and restores registers. **With communication lost
  torque is left ON** (the servos hold the last goal): dropping the arm is worse than holding it.
- Ctrl-C only sets a flag; the loop stops at the next tick through the same path.

### 6.3 Findings that changed the design (each has a regression test)
1. **The safety Torque_Limit changes the plant.** At 35% the servo clamps at ~1.2 N.m instead of 3.35, so the same excitation
   saturates most of the time. The recorder stores `torque_scale` per joint in every log; the fit scales `forcerange` to match
   (`model.scale_force_limits`) and pre-flight simulates the reduced limit. *Assumes* Torque_Limit scales max torque linearly (Phase 6).
2. **The goal's acceleration cap must not shape the excitation.** Armature is only visible when the goal changes faster than the
   servo can accelerate (~Tmax/J, about 150 rad/s^2). A 20 rad/s^2 cap (the first bring-up guess) made steps unsaturating and armature
   unidentifiable (fit pinned it at its bound). Default is now 400 rad/s^2 (a 0 -> 2 rad/s change in one 10 ms tick); safety rests on
   the rate cap, soft limits, torque limit and tracking watchdog.
3. **Sub-step command timing was a systematic misfit.** MuJoCo applies one control value per 2 ms step. Sampling the zero-order hold at
   the step start moves every command edge to the next boundary (uniform 0-2 ms, arbitrary phase against the log clock); parameters
   absorbed it (armature to its bound, delay +2 ms). `data.step_averaged_hold` averages the hold over each step. Phase 2 missed this
   because its mock plant shared the fit's grid ("inverse crime"); the bus plant has an independent clock.
4. **A finer plant than the fit model is not recoverable as "truth".** With a 0.5 ms plant and the 2 ms fit model the integrator
   error is absorbed by the parameters (armature 0.3x). That is what "make *this* simulation match" should do, so the calibrated
   values are effective values *for the MJCF's timestep*; the mock bus uses the model's own step so parameters can be compared.
5. **Confidence intervals are not trustworthy when the actuator is saturated or sub-tick accurate.** FD Jacobians there are
   non-smooth: kp reported "+-0%" while 40% off. Rules now: CIs come from the full (unpinned) fit; flagged parameters are pinned only
   to produce a self-consistent exported set; and the arbiter of a calibration is **held-out prediction error**
   (`optimizer.validate_joint`), not parameter agreement. kp, kv and friction are weakly observable at 1.5 mrad encoder resolution
   against a 3.4 mrad linear band; damping, armature and delay are what the data pins down.
6. Mock-bus read sampling returned a state up to one step stale (+1 ms apparent delay); fixed by stepping one `dt` past the sample.

### 6.4 Result of the record -> fit -> validate chain on the mock bus
`tests/test_pipeline_e2e.py` (slow, ~90 s): plan -> pre-flight -> safety session -> recorder -> MockBus (26 ms delay, 12-bit
quantization, 0.3-tick read noise, wake-up jitter, torque limit 0.7) -> fit on stage 2 -> validate on the held-out stage-3 multisine.
Measured: cost 1.26x the quantization floor, delay 26.20 ms (truth 26.0), **held-out RMS 0.55 mrad vs 2.7 mrad for the nominal model**
(4.9x better, about a third of one encoder tick). Parameters vs truth: kv 0.99, damping 0.90, armature 0.89, kp 1.31, friction 0.21.
Friction is the one the CI got wrong (it was not flagged): its 0.05 N.m is invisible next to a 3.35 N.m clamp, which is exactly
why held-out error, not parameter agreement, is the acceptance criterion.

## 6. Phase 6 findings (real SO-101, 12 V, one arm)

Full analysis in [POSTMORTEM.md](POSTMORTEM.md). Interface facts only:

- Stable port: use the udev alias (`/dev/so101-follower`), never `ttyACM*`; the numbering changes when other USB
  serial devices are plugged in.
- Bus: 1 Mbps, six-servo sync read 1.35-1.5 ms, 100 Hz loop with 0.05 ms jitter.
- `Present_Velocity` is in ticks/s with the correct sign. `Present_Load` is non-zero and signed only with torque on
  (about +-30 to 140 raw); its physical unit was not established.
- About 0.3% of reads return a corrupt temperature or voltage byte. Debounce before acting on them.
- A register write can lose its status packet (seen once on torque enable). Retry; writes are idempotent.
- Releasing torque lets a gravity-loaded arm fall (shoulder_lift sagged about 47 degrees). Hold torque at the end of a
  run and release only when the arm is supported (`autosysid release`).
- Registers read at default: P=16, I=0, D=32, Minimum_Startup_Force=16, dead zones 1, Torque_Limit 1000 (gripper 500).
- Unresolved: the Phase 0 note says MJCF zero is mid-range, but the code uses tick 2047. Differs by up to about 9
  degrees on some joints. Not checked further because the project was stopped.
- Not verified: whether `Torque_Limit` scales torque linearly.
