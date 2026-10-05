# Experiment log

Chronological record of jobs, settings, results and decisions. Times are Jubail local (GST, UTC+4) unless noted.
Every final-test (official LIBERO init states) run must be listed in the "Test-set access" table.

## Test-set access

| date | policy / arch | ckpt | job | note |
|---|---|---|---|---|

## Log

### 2026-10-05 — setup
- Repo initialised (local + GitHub `Autzoko/RoboticsNAS`, HPC checkout at `/scratch/ll5582/RoboticsNAS`).
- LeRobot pinned at `3f2c29e` (`third_party/lerobot`), env `env/` built by job 18656149 (`scripts/setup_env.sbatch`).
- Dataset: local `lerobot/libero` snapshot at `/scratch/ll5582/LIBERO/lerobot_libero` (HF revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`).
- Env fixes: gcc module needed for `egl_probe` build; `opencv-python` replaced by headless (no libGL on nodes);
  project-local LIBERO config (`configs/libero/config.yaml`, `LIBERO_CONFIG_PATH`) because `~/.libero` points to a
  non-existent path; LIBERO assets downloaded from HF Hub into `cache/libero/assets` (+ package symlink).
- Pins (`configs/pins.json`): smolvla_base `d9f33c94`, SmolVLM2-500M-Video-Instruct `7b375e1b`.
- Split (`configs/split.json`): 3 demos/task held out as offline-val (120 eps), 1573 train eps; stats on train only.

### 2026-10-05 — E0 correctness (jobs 18656389, 18656437)
- smolvla_base VLM layers 0-15 are bit-identical to SmolVLM2 (max diff 0) -> VLM was frozen in pretraining;
  appending SmolVLM2 layers 16-23 for the `n_vlm>16` axis is consistent.
- Elastic forward at default arch vs LeRobot SmolVLA (fp32): train-loss max abs diff 5.7e-6 (mean 0.377);
  `sample_actions` max abs diff 0. FFN neuron sorting function-preserving (5e-7).
- LIBERO (EGL, 256x256, hard reset) runs on A100 nodes. Common random numbers verified: identical reset seeds give
  identical sim-state hashes across archs; different episodes give different layouts.
- Untrained (smolvla_base, no LIBERO training) SR = 0/2 on spatial task 0, as expected. 280-step batch: 15-31 s.
