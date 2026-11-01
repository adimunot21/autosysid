# Committed hardware logs

Raw `autosysid record` output (`RawLog` `.npz`, 100 Hz, all 6 joints per file, targets in ticks-quantized radians),
one folder per excited joint. `scripts/reproduce_postmortem.py` and `scripts/validate_wrapper.py` read them recursively.

| Folder | Source run | Joints excited | Stage-3 holdout |
|---|---|---|---|
| `wrist_roll/` | `runs/stage1_wrist_roll`, `runs/stage2_wrist_roll` | wrist_roll | no (steps segment is held out) |
| `shoulder_pan/` | `runs/multi_shoulder_pan` | shoulder_pan | no |
| `wrist_flex/` | `runs/multi_wrist_flex` | wrist_flex | no |

`shoulder_lift`, `elbow_flex` and `gripper` have no data yet. Recorded 12 V, default servo PID (P=16 I=0 D=32),
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
