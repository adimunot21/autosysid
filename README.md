# AutoSysID

Hardware-in-the-loop system identification for serial-bus robot arms. Plug in an SO-101 (Feetech STS3215), run a safe
excitation sequence, and get a calibrated MuJoCo MJCF. The optimization is DeepMind's `mujoco.sysid`; AutoSysID provides
the hardware wrapper, safety layer, data plumbing, honest uncertainty reporting and the XML export.

**Status: feature-complete on a simulated arm; not yet run on real hardware** (Phase 5 of 6). Everything that touches a
servo has been tested against a physics-backed mock bus and a fake SDK only. Read "Known limitations" before trusting a
result from real hardware.

## Why

A MuJoCo model of the SO-101 ships with generic servo numbers. A real arm has its own friction, inertia and, above all, a
command-to-motion latency of tens of milliseconds. Policies trained against the generic model transfer poorly. AutoSysID
measures those numbers on *your* arm and writes them into your own MJCF.

## Try it without hardware (2-3 minutes)

```bash
conda create -n autosysid python=3.12 && conda activate autosysid
pip install -e ".[dev]"
autosysid info
autosysid demo
```

`demo` simulates an arm whose dynamics are secretly different from its XML, records the full protocol through the same
safety layer and recorder used on hardware, fits, validates on held-out motion, exports, and prints the recovered values
next to the truth.

## On the arm

```bash
autosysid doctor                                  # read-only: ping, pose, temperature, voltage. Moves nothing.
autosysid calibrate --xml scene.xml --stages 1    # first time: smallest motions only, confirms before moving
autosysid calibrate --xml scene.xml               # all three stages -> scene_calibrated.xml (+ .calibration.json)
```

Or step by step, which also lets you re-fit without the arm: `record` -> `fit` -> `validate` -> `export` (see `--help`).
Run directories (`runs/<timestamp>/`) hold the raw logs, fits and validation so any result can be reproduced.

Before the arm moves you see a simulation pre-flight (predicted tracking error, actuator saturation, contacts, limits) and
must confirm. While it moves: joint limits at 80% of the calibrated range, rate/acceleration limiter, watchdogs on
temperature, voltage, tracking error, loop timing and communication, and Ctrl-C triggers the same safe stop. A trip returns
the arm to its start pose slowly, then releases torque and restores the servo registers it changed. If communication is
lost the torque is left on, on purpose.

**Registers written while recording:** `Torque_Limit` (reduced to 70% of its value by default, `--torque-limit`) and
nothing else that persists; it is restored afterwards. No EEPROM register is reachable through the bus API.

## How it works

```
plan (3 stages) -> pre-flight (simulation) -> safety session -> recorder (100 Hz, sync read/write)
   -> raw logs -> per-joint fit with mujoco.sysid (kp, kv, damping, friction, armature + latency)
   -> held-out validation -> patch the MJCF in place -> calibration.json
```

- **Stage 1** small steps/ramps per joint (checks the data path, latency). **Stage 2** per joint: slow ramps, an unsaturated
  chirp and slew-limited saturating steps (each exposes different parameters). **Stage 3** all joints multisine; one segment
  is held out and never fitted.
- The exporter edits only the start tags of the target `<joint>` and `<position>` elements, so comments, includes and the
  shared `<default class>` stay as they were. It re-compiles the result and reads every value back before reporting success.
- Parameters the data cannot pin down are reported as such and are **not** written; the arbiter of a calibration is the
  prediction error on motion it never saw (`validate`).

## Known limitations (read these)

- **Unverified on a real arm:** the tick-to-radian zero (assumed to be 2047 = LeRobot's calibration pose), that
  `Torque_Limit` scales torque linearly, and the units of velocity/load readings. Phase 6 checks each of these.
- **kp, kv and friction are weakly observable.** Encoder resolution (1.5 mrad) is comparable to the servo's linear band
  (about 3 mrad), so these mostly do not affect the data; damping, armature and latency are what is identified. Confidence
  intervals for the weak ones can be wrong when the actuator saturates (friction came back 5x low without being flagged in
  the end-to-end test), which is why held-out error decides acceptance.
- **Calibrated values are effective values for the MJCF's timestep** (default 2 ms), not physical constants.
- Fits take about 1-2 minutes per joint on the 3-joint test model; the full SO-101 mesh model is untested for speed.
- Only joints with a `<position>` actuator of the same name are supported. Dynamixel and other buses are not implemented.

## Development

```bash
pytest                   # fast suite (~20 s)
pytest -m slow           # end-to-end fits (~90 s each)
ruff check . && ruff format .
```

Design notes, every external-interface finding and every trap found along the way are in `docs/data-contract.md`.
