# Committed hardware logs

Raw `autosysid record` output (`RawLog` `.npz`, 100 Hz, all 6 joints per file, targets in ticks-quantized radians),
one folder per excited joint. `scripts/reproduce_postmortem.py` and `scripts/validate_wrapper.py` read them recursively.

| Folder | Source run | Joints excited | Stage-3 holdout |
|---|---|---|---|
| `wrist_roll/` | `runs/stage1_wrist_roll`, `runs/stage2_wrist_roll` | wrist_roll | no (steps segment is held out) |
| `shoulder_pan/` | `runs/multi_shoulder_pan` | shoulder_pan | no |
| `wrist_flex/` | `runs/multi_wrist_flex` | wrist_flex | no |
| `shoulder_lift/pose1/` | `runs/hw/shoulder_lift_pose1_a1` (stage 1), `_s2` (stage 2), 2026-11-01 | shoulder_lift, base pose 2045 ticks, `--behavior-sweep --amplitude-scale 1.0` | no |
| `shoulder_lift/pose2/` | `runs/hw/shoulder_lift_pose2_s1`, `_s2`, 2026-11-01 | shoulder_lift, base pose 1475 ticks (-50 deg), same flags | no |
| `shoulder_lift/pose3/` | `runs/hw/shoulder_lift_pose3b_s1` (scale 0.8), `_s2` (scale 0.8), 2026-11-01 | shoulder_lift, base pose 2581 ticks (+47 deg) with elbow folded to 1586 ticks to clear the table | no. An earlier pose-3 attempt (elbow at 0 deg) hit the table and was discarded. |
| `shoulder_lift/pose4/` | `runs/hw/shoulder_lift_pose4_s1`, `_s2`, 2026-11-01 | shoulder_lift, base pose 1914 ticks (-13 deg), elbow/wrist as in poses 1-2; reached with `autosysid move` | no. Held out of the poses 1-3 fit. |

`elbow_flex` and `gripper` have no data yet; `shoulder_lift` has one base pose (a first stage-1 run at scale 0.5 barely moved the joint and was not kept). Recorded 12 V, default servo PID (P=16 I=0 D=32),
Torque_Limit 70%, one arm. `record*.json` next to the logs is the recorder's own provenance (config, segment names).

## Adding a joint or a new base pose (arm supported, hand on the power switch)

    autosysid doctor                                              # read-only
    python scripts/print_gravity_poses.py --xml <scene.xml>       # suggested base poses (read-only)
    # place the arm at the base pose by hand, then:
    autosysid record --xml <scene.xml> --joints shoulder_lift --behavior-sweep --stages 1,2,3 \
        --run-dir runs/hw/shoulder_lift_pose1
    mkdir -p hardware_logs/shoulder_lift/pose1 && cp runs/hw/shoulder_lift_pose1/raw/*.npz hardware_logs/shoulder_lift/pose1/

Repeat with a different base pose (`pose2`, `pose3`) for each gravity-loaded joint: a cosine gravity term is only
identifiable from sweeps at several configurations. Torque is held at the end of every run; use `autosysid release`
only once the arm is supported. A stage-3 run records a `role="holdout"` multisine, which the evaluation prefers.
