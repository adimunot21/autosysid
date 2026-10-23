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
