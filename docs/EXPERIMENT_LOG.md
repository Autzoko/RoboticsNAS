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
