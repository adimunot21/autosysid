# AutoSysID

Hardware-in-the-loop system identification for serial-bus robot arms. Plug in an SO-101 (Feetech STS3215),
run a safe excitation sequence, and get a calibrated MuJoCo MJCF. The optimization is DeepMind's `mujoco.sysid`;
AutoSysID provides the hardware wrapper, safety layer, data plumbing and XML export.

**Status: under construction** (Phase 3 of 6). See `docs/data-contract.md` for what has been verified so far.

```bash
conda create -n autosysid python=3.12 && conda activate autosysid
pip install -e ".[dev]"
autosysid info
pytest
```
