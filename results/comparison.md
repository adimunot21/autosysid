Held-out RMS position error, encoder ticks (1 tick = 1.534 mrad)

| Joint | Hold still | Stock MJCF | Stock + best delay | mujoco.sysid* | 3-param | 4-param | 4-param in MJCF | Stock / 4-param | (Stock+delay) / 4-param |
|---|---|---|---|---|---|---|---|---|---|
| shoulder_pan | 72.07 | 16.70 | 4.63 | n/a | 2.39 | 2.36 | 2.56 | 7.1x | 2.0x |
| wrist_flex | 69.67 | 16.94 | 7.96 | n/a | 4.89 | 2.15 | 2.54 | 7.9x | 3.7x |
| wrist_roll | 71.08 | 16.16 | 5.81 | 2.27 | 2.22 | 2.14 | 2.54 | 7.6x | 2.7x |

Held-out = stage-3 multisine where recorded, else the stage-2 'steps' segment (see evaluation.split_logs).
'4-param in MJCF' = the 4-parameter model in front of an ideal-actuator copy of the stock MJCF (sim_wrapper).
*mujoco.sysid is the wrist_roll number from results/sysid_reference.json; its split was not saved, so it is not directly comparable (re-run with --rerun-sysid for a like-for-like value).
