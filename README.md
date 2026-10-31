# autosysid

A tool to identify an SO-101 robot arm automatically and write the results into a MuJoCo model. The arm moves on its own, and friction, damping, armature and servo gains are fitted from the recorded motion.

The pipeline works on a simulated arm. On the real arm it fails to separate the physical parameters, and the project is archived. The reasons are in [docs/POSTMORTEM.md](docs/POSTMORTEM.md).

## Why it failed on hardware

The servo's internal position loop hides almost all of the arm's dynamics. It never saturated in any test, so the recorded data supports only a delay, a first-order lag, a deadband and a gravity offset for each joint. `mujoco.sysid` fitted worse than a three-parameter model on held-out data. Tested joints: wrist_roll, shoulder_pan and wrist_flex. Untested: elbow_flex, shoulder_lift and gripper. The final check, whether the fitted model brings the simulation closer to the real arm, was not done.

## What works

- `src/autosysid/hardware/`: Feetech STS3215 bus driver, a safety layer (soft limits, rate limits, watchdogs, safe stop) and a 100 Hz recorder. All of it ran on a real arm.
- `autosysid doctor`: read-only health check of the arm. Nothing moves.
- `autosysid release`: turns torque off once the arm is supported.
- `docs/data-contract.md`: what the bus, SDK and registers return, including quirks such as corrupt temperature and voltage bytes, lost status packets, and the arm sagging when torque is released.
- The fitting and export code runs and is tested. On real hardware its physical parameters are unreliable.

## Running

    conda create -n autosysid python=3.12 && conda activate autosysid
    pip install -e ".[dev]"
    autosysid demo       # simulated arm, no hardware
    autosysid doctor     # real arm, read-only

`autosysid record` moves the arm. Keep a hand on the power switch and read `--help` first. Torque stays on at the end of a run by default, because releasing it made the shoulder drop about 47 degrees. Support the arm before running `release`.

## Tests

    pytest              # fast suite
    pytest -m slow      # end-to-end fits on the simulated arm
