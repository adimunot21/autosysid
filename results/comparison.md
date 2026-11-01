Held-out RMS position error, encoder ticks (1 tick = 1.534 mrad)

| Joint | Hold still | Stock MJCF | Stock + best delay | mujoco.sysid* | 3-param | 4-param | 5-param | 4-param in MJCF | Stock / wrapped | (Stock+delay) / wrapped |
|---|---|---|---|---|---|---|---|---|---|---|
| elbow_flex | 83.30 | 23.56 | 13.82 | n/a | 12.02 | 4.28 | 3.18 | 3.72 | 6.3x | 3.7x |
| gripper | 71.67 | 16.16 | 5.76 | n/a | 2.48 | 2.42 | n/a | 2.58 | 6.3x | 2.2x |
| shoulder_lift | 78.81 | 27.80 | 19.74 | n/a | 21.38 | 5.78 | 4.50 | 4.72 | 5.9x | 4.2x |
| shoulder_pan | 72.07 | 16.70 | 4.63 | n/a | 2.39 | 2.36 | n/a | 2.56 | 6.5x | 1.8x |
| wrist_flex | 69.67 | 16.94 | 7.96 | n/a | 4.89 | 2.15 | n/a | 2.54 | 6.7x | 3.1x |
| wrist_roll | 71.08 | 16.16 | 5.81 | 2.27 | 2.22 | 2.14 | n/a | 2.54 | 6.4x | 2.3x |

Held-out = stage-3 multisine where recorded, else the stage-2 'steps' segment (see evaluation.split_logs); for the multi-pose joints the steps segment of every base pose is held out.
'4-param in MJCF' = the richest fitted model (5-param where shown) in front of an ideal-actuator copy of the stock MJCF (sim_wrapper). 4-param uses g0*cos(q - q_horiz) on shoulder_lift and elbow_flex (q_horiz from the MJCF) and a constant offset elsewhere; 5-param adds d0 + d1*|cos| where >= 3 base poses exist.
*mujoco.sysid is the wrist_roll number from results/sysid_reference.json; its split was not saved, so it is not directly comparable (re-run with --rerun-sysid for a like-for-like value).
