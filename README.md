# AutoSysID

A tool I built to identify an SO-101 arm and write the results into a MuJoCo model. The idea was to plug the arm in,
let it move on its own, and get friction, damping, armature and servo gains out. It does not work on the real arm,
and I stopped.

The short version is in [docs/POSTMORTEM.md](docs/POSTMORTEM.md). Read that first.

## What happened

The pipeline works on a simulated arm. On a real SO-101 the servo's own position loop hides almost everything.
It never saturated in any test, so the data only supports a delay, a first-order lag, a deadband and a gravity offset
per joint. The physical parameters can't be separated. `mujoco.sysid` fit worse than a three-parameter model on
held-out data. Tested on wrist_roll, shoulder_pan and wrist_flex. Not tested: elbow_flex, shoulder_lift, gripper.
I also never checked whether the result makes a simulation closer to the real arm.

## What is in here that still works

- `src/autosysid/hardware/`: Feetech STS3215 bus driver, safety layer (soft limits, rate limits, watchdogs, safe
  stop), a 100 Hz recorder. Used on a real arm.
- `autosysid doctor`: read-only check of the arm. Moves nothing.
- `autosysid release`: turns torque off after you've supported the arm.
- `docs/data-contract.md`: what the bus, SDK and registers actually return, including the quirks (corrupt
  temperature/voltage bytes, lost status packets, torque release letting the arm sag).

The fit and export code runs and is tested, but its output on real hardware isn't trustworthy for the physical
parameters.

## Running it

```bash
conda create -n autosysid python=3.12 && conda activate autosysid
pip install -e ".[dev]"
autosysid demo            # simulated arm, no hardware
autosysid doctor          # real arm, read-only
```

`record` moves the arm. Keep a hand on the power switch, support the arm when you run `release`, and read `--help`
first. At the end of a run the torque stays on by default, because letting go made the shoulder drop about 47 degrees.

## Tests

```bash
pytest                    # fast suite
pytest -m slow            # end-to-end fits on the simulated arm
```

## Status

Archived. Not planning more work on it unless someone shows the downstream benefit.
