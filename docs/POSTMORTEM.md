# AutoSysID post-mortem

Written for a reader (human or LLM) who is about to propose or build "plug in an SO-101, get its physical parameters
into a MuJoCo file". We built that. It does not work on the real arm. This is what was tried and measured.

## What we set out to do

Plug in an SO-101 arm (Feetech STS3215 servos). Run an automatic, safe motion sequence. Fit the logged data with
DeepMind's `mujoco.sysid`. Get a calibrated MJCF with per-joint kp, kv, damping, frictionloss, armature, plus
command-to-motion latency.

## What happened

1. Phases 0-5: built the CLI, safety layer, recorder, fitter and MJCF exporter. All tested on a simulated arm. On that
   simulated arm the pipeline recovers the parameters. That test is circular: the simulated arm has the same model
   structure as the fitted model, so it says nothing about a real servo.
2. Phase 6: ran it on a real arm. Joints tested: wrist_roll, shoulder_pan, wrist_flex.
   Joints not tested: elbow_flex, shoulder_lift, gripper.

## What worked

- Talking to the arm: 1 Mbps, 6 servos read in about 1.4 ms, 100 Hz loop with 0.05 ms jitter.
- A safety layer that held up on hardware (soft limits, rate limits, watchdogs, safe stop).
- Finding real bus problems: about 0.3% of reads have a corrupt temperature or voltage byte (false over-temperature
  trip), and a torque-enable write can lose its status packet. Both were fixed (debounce, retries).
- Velocity units (ticks/s, correct sign) and joint directions confirmed.

## What did not work

**The physical parameters cannot be identified.** The servo's internal position loop is much faster than anything the
excitation can expose. In every run the servo never saturated: peak velocity about 1 rad/s and acceleration about
20 rad/s^2, against roughly 120-150 rad/s^2 in the model. The response is dominated by one first-order lag, a small
deadband and a short delay. kp, kv, damping, friction and armature enter only as combinations. For a mostly
P-controlled joint: lag time constant is about (kv + damping) / kp and deadband is about friction / kp. Nothing in the
data separates them. Any single value reported for them would be a guess dressed as a measurement.

**The `mujoco.sysid` fit lost to a three-parameter baseline.** Held-out prediction error on wrist_roll (position error,
encoder ticks, lower is better):

| Model | Error |
|---|---|
| Hold still | 9.35 |
| Stock MJCF | 5.36 |
| Best `mujoco.sysid` fit (6 starts, wide bounds) | 2.27 (cost 32x the noise floor; kp and damping at their bounds) |
| delay + first-order lag + deadband | 1.35 (fit in under a minute) |

The first sysid fit was rejected outright at 186x the noise floor. Its 70 ms delay was an artifact: deadband and lag
were being absorbed into dead time. The simple model gives 10-22 ms.

**Same result on three joints** (held-out error in ticks; the 4-parameter model adds a constant offset for gravity):

| Joint | 3-param | 4-param | Lag | Delay | Deadband | Offset |
|---|---|---|---|---|---|---|
| wrist_roll | 1.43 | 1.20 | 69 ms | 10 ms | 3.9 ticks | -0.5 |
| shoulder_pan | 1.37 | 1.32 | 57 ms | 22 ms | 3.5 ticks | +0.5 |
| wrist_flex | 4.93 | 1.20 | 58 ms | 18 ms | 6.0 ticks | -4.2 (gravity) |

So what the arm actually exposes is four numbers per joint: lag, delay, deadband, gravity offset. These can be
exported to an MJCF through a closed-form mapping, but only as ratios, not as physical constants.

## What was never tested (so do not read more into this than is there)

- elbow_flex and shoulder_lift (the most gravity-loaded joints), and the gripper.
- Any other arm, other servo firmware settings, or a 7.4 V supply. One arm, 12 V, default PID (P=16, I=0, D=32).
- Changing the servo's P gain or operating mode to expose the dynamics. That might make more parameters visible, but the
  result would describe a re-tuned arm, not the stock one.
- Whether a policy or trajectory behaves closer to the real arm in a simulation calibrated with these four numbers. This
  is the only test that decides if the output is useful, and it was not run.
- Prior art. Existing SO-101 or Feetech identification work was not searched for.

## Why we are stopping

- The advertised output (friction, damping, armature, gains in an MJCF) is not recoverable from this arm in position
  mode. This is a property of the servo's control loop, not of our code, and it should hold for other stock SO-101s.
- What is recoverable is four numbers per joint. A short script gets them, and the safe recorder does not need a
  solver library.
- Nothing shows those four numbers improve sim-to-real transfer. That was the claim that would have justified the project.
- The usefulness of the leftover pieces (safe recorder, bus-fault handling) is real but small and generic.

## If an LLM suggests this project to you

The common suggestion is some version of "excite each joint, fit friction/damping/inertia, write them to the MJCF".
For a position-controlled STS3215 that is the part that fails. Before adopting it, ask:

1. Does the servo ever leave its linear region during the excitation? Here it did not.
2. Are the parameters separately identifiable, or only their ratios? Check confidence intervals, not just fit cost.
3. Does the fit beat a delay + lag + deadband model on held-out data? Here it did not.
4. Does a downstream task (policy rollout, trajectory tracking) improve? Test this before building anything else.

## Repository

Code is kept as is, with 166 passing fast tests. Run data is not committed. See README.md for usage and
`docs/data-contract.md` for every bus and SDK finding.

## Addendum (2026-11-01): the behavioral model, reproduced and wrapped

The four-number model is now code (`src/autosysid/behavior.py`) with tests, a MuJoCo wrapper
(`src/autosysid/sim_wrapper.py`, `STS3215SimWrapper`) and a reproduction script (`scripts/reproduce_postmortem.py`)
that reads the committed logs in `hardware_logs/`. Held-out RMS error in ticks (held out = the stage-2 "steps"
segment, fitted on the rest; a fresh stage-3 holdout is still to be recorded):

| Joint | Hold still | Stock MJCF | Stock + best delay | 3-param | 4-param | 4-param in MJCF |
|---|---|---|---|---|---|---|
| shoulder_pan | 72.1 | 16.7 | 4.6 | 2.39 | 2.36 | 2.56 |
| wrist_flex | 69.7 | 16.9 | 8.0 | 4.89 | 2.15 | 2.54 |
| wrist_roll | 71.1 | 16.2 | 5.8 | 2.22 | 2.14 | 2.54 |

What this changes about the original claims:

- **The "1.20 ticks" is not a held-out number.** Fitting on all logs gives 1.2-1.5 ticks in-sample. On held-out data the
  error is 2.1-2.4 ticks, and 0.98-1.7 when the slow ramp is held out but 2.7-2.9 when the chirp is. The model is
  weakest at higher frequency, which is what a 1.5-2 Hz feedforward will exercise.
- **Delay is not identifiable at 100 Hz.** The fitted Td for the same joint moves between 4 and 16 ms depending on
  which segment is held out (wrist_roll); tau (50-67 ms) and the deadband (4-6 ticks) are stable.
- **The "5.36 ticks stock" figure is not reproduced.** A plain stock-MJCF rollout scores about 16-17 ticks, the same as
  using the command as the position, because the stock actuator has no lag. 5.36 must have included a fitted delay
  and an excluded start window (mujoco.sysid does both). Against *stock + best delay* (the fair baseline) the
  4-parameter model is 2.0-3.7x better, against plain stock 7-8x.
- **Gravity matters on wrist_flex only** among the three joints tested (4.89 -> 2.15 ticks with a constant offset). The
  cosine form `g0*cos(q - q_horiz)` could not be tested: shoulder_lift, elbow_flex and gripper have no data yet.
  `q_horiz` comes from the MJCF (`src/autosysid/gravity.py`; -142 deg and -178 deg for shoulder_lift and elbow_flex on the
  vendored model), not from the fit, because a single small sweep cannot separate it from `g0`.
- **The stock actuator must not sit under the filter.** Its forcerange saturates on a 0.5 rad step (108 ticks of error)
  and adds ~8 ms of lag; `ideal_actuator_model` removes both and is required for the wrapper.
- `mujoco.sysid` (wrist_roll, 2.27 ticks) remains the only reference number; its split was not saved, so it is
  carried as a flagged value in `results/sysid_reference.json`.

### Addendum 2 (2026-11-01): shoulder_lift, three base poses

shoulder_lift was recorded at base poses +4 deg, -50 deg and +47 deg (`hardware_logs/shoulder_lift/pose{1,2,3}`; an
earlier +47 deg attempt with the elbow straight hit the table and was discarded).

- **Gravity is the dominant effect on this joint and the cos form is right.** Sag at the three poses is about +19, +2 and
  +24-29 ticks. `g0 = -23 ticks` fitted from poses 1+2 alone predicts pose 3 at +22.9; `q_horiz = -142 deg` comes
  from the MJCF, not the fit. A constant offset fitted on one pose is worse than no model on another (18.7 vs 16.7 ticks).
- **The deadband is not constant.** Fitted per pose: d = 3.9 / 14.0 / 19.2 ticks at |cos(q - q_horiz)| = 0.04 / 0.82 / 0.99.
  `BehaviorParams.deadband_gravity_rad` (d1) adds `d(q) = d0 + d1*|cos|`, optional and off by default (old
  parameter files load unchanged).
- **Leave-one-pose-out (fit on two poses, score the third, no refit):**

| Held-out pose | 4-param, shared d | 5-param, d0 + d1 abs(cos) | target as position |
|---|---|---|---|
| +4 deg | 5.81 | 5.85 | 27.4 |
| -50 deg | 9.44 | 4.79 | 16.7 |
| +47 deg | 9.43 | 6.61 | 33.2 |

  In-sample, all three poses: 6.07 vs 3.91 ticks. The -50 deg case is extrapolation to the unloaded joint, and `d0`
  hits its zero bound when both training poses are heavily loaded; a fourth pose near |cos| = 0 would constrain it.
- Fit with `autosysid fit-behavior ... --gravity-xml scene.xml --load-dependent-deadband`.
- Delay (20-38 ms) and tau (33-53 ms) remain loosely identified across poses and splits.

**Held-out pose (-13 deg, |cos| = 0.67, reached with `autosysid move`).** Fit on poses 1-3 only, no refit: 3.01 ticks for the
5-parameter model (4-parameter shared-d: 3.19; 3-parameter, no gravity: 15.95; target as position: 24.9; pose 4's own
fit: 2.55). The fit on poses 1-3 predicted d = 13.0 ticks here and fitting pose 4 alone gives 12.6. On all four
poses together: delay 26.6 ms, tau 47.5 ms, d0 4.0 + d1 13.3 ticks, g0 -25.7 ticks (`results/shoulder_lift_5param_4poses.json`).
Pose 4 is close in load to poses 1 and 3, so it does not separate the 5- from the 4-parameter model;
the leave-pose-2-out result above (9.4 -> 4.8 ticks) is the evidence for d1.
